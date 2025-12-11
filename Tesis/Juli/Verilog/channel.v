`timescale 1ns/1ps

module channel_with_noise #(
    parameter DWIDTH      = 16, // Ancho de datos de la señal 
    parameter CWIDTH      = 8,  // Ancho de coeficientes S(8,6)
    parameter NOISE_WIDTH = 16, // Salida del IP Core S(16,11)
    parameter SNR_WIDTH   = 11  // Factor de escala SNR S(11,10)
)(
    input  wire                         clk,
    input  wire                         rst,
    
    // --- Entradas de Señal (Compleja) ---
    input  wire signed [DWIDTH-1:0]     In_I, // Señal de entrada Real
    input  wire signed [DWIDTH-1:0]     In_Q, // Señal de entrada Imag
    
    // --- Configuración ---
    // Coeficientes para el canal complejo h = h_R + j h_I
    input  wire signed [CWIDTH-1:0]     h_real_coeff_in, 
    input  wire signed [CWIDTH-1:0]     h_imag_coeff_in,
    input  wire                         coeff_wr_en, // Habilitador para cargar coeficientes
    
    // Factor "Sigma" para ajustar la SNR
    input  wire signed [SNR_WIDTH-1:0]  sigma_scale, 
    
    // --- Salidas (Compleja con Ruido) ---
    output wire signed [DWIDTH-1:0]     Out_I,
    output wire signed [DWIDTH-1:0]     Out_Q
);

    // ================================================================
    // 1. LOS 4 FILTROS 
    //    Implementa la convolución compleja: (I+jQ)*(hR+jhI)
    // ================================================================
    
    wire signed [DWIDTH-1:0] y_ii, y_qq, y_iq, y_qi;

    // Instancia 1: Camino I -> I (Usa h_Real) -> Parte del término Real
    filtro_fir #(.H(11), .W(DWIDTH)) u_fir_ii (
        .clk      (clk),
        .rst      (rst),
        .din      (In_I), 
        .coeff_in (h_real_coeff_in), 
        .wr_en    (coeff_wr_en),
        .dout     (y_ii)
    );

    // Instancia 2: Camino Q -> Q (Usa h_Imag) -> Parte del término Real (Negativo)
    filtro_fir #(.H(11), .W(DWIDTH)) u_fir_qq (
        .clk      (clk),
        .rst      (rst),
        .din      (In_Q), 
        .coeff_in (h_imag_coeff_in), 
        .wr_en    (coeff_wr_en),
        .dout     (y_qq)
    );

    // Instancia 3: Camino I -> Q (Usa h_Imag) -> Parte del término Imag
    filtro_fir #(.H(11), .W(DWIDTH)) u_fir_iq (
        .clk      (clk),
        .rst      (rst),
        .din      (In_I), 
        .coeff_in (h_imag_coeff_in), 
        .wr_en    (coeff_wr_en),
        .dout     (y_iq)
    );

    // Instancia 4: Camino Q -> I (Usa h_Real) -> Parte del término Imag
    filtro_fir #(.H(11), .W(DWIDTH)) u_fir_qi (
        .clk      (clk),
        .rst      (rst),
        .din      (In_Q), 
        .coeff_in (h_real_coeff_in), 
        .wr_en    (coeff_wr_en),
        .dout     (y_qi)
    );

    // ================================================================
    // 2. COMBINACIÓN (Sumadores/Restadores)
    //    Matemática: (I*hR - Q*hI) + j(I*hI + Q*hR)
    // ================================================================
    reg signed [DWIDTH-1:0] channel_out_I;
    reg signed [DWIDTH-1:0] channel_out_Q;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            channel_out_I <= {DWIDTH{1'b0}};
            channel_out_Q <= {DWIDTH{1'b0}};
        end else begin
            // Parte Real: Resta (por j*j = -1)
            channel_out_I <= y_ii - y_qq; 
            // Parte Imaginaria: Suma
            channel_out_Q <= y_iq + y_qi; 
        end
    end
    
    // ================================================================
    // 3. GENERADOR DE RUIDO Y ESCALADO (Sigma)
    // ================================================================
    
    // Cables para salida cruda del IP Core
    wire signed [NOISE_WIDTH-1:0] raw_noise_I;
    wire signed [NOISE_WIDTH-1:0] raw_noise_Q;
    
    // Instancia del IP Core (Nombre genérico, se ajustara cuando tengamos las cosas)
    gng_core u_noise_gen (
        .clk         (clk),
        .rst         (rst),
        .noise_out_1 (raw_noise_I),
        .noise_out_2 (raw_noise_Q) 
    );

    // --- Multiplicación por Sigma ---
    // Resultado crece a (NOISE_WIDTH + SNR_WIDTH) bits
    wire signed [NOISE_WIDTH + SNR_WIDTH - 1 : 0] mult_res_I;
    wire signed [NOISE_WIDTH + SNR_WIDTH - 1 : 0] mult_res_Q;

    assign mult_res_I = raw_noise_I * sigma_scale;
    assign mult_res_Q = raw_noise_Q * sigma_scale;

    // Seleccionamos los bits más significativos válidos para volver a 16 bits.
    wire signed [DWIDTH-1:0] final_noise_I;
    wire signed [DWIDTH-1:0] final_noise_Q;
    
    assign final_noise_I = mult_res_I[NOISE_WIDTH + SNR_WIDTH - 2 -: DWIDTH];
    assign final_noise_Q = mult_res_Q[NOISE_WIDTH + SNR_WIDTH - 2 -: DWIDTH];

    // ================================================================
    // 4. SUMA FINAL CON SATURACIÓN 
    //    Evita overflow si Señal + Ruido > Max Valor posible
    // ================================================================
    
    // Usamos 1 bit extra para detectar el desbordamiento
    wire signed [DWIDTH:0] sum_temp_I;
    wire signed [DWIDTH:0] sum_temp_Q;

    assign sum_temp_I = {channel_out_I[DWIDTH-1], channel_out_I} + {final_noise_I[DWIDTH-1], final_noise_I};
    assign sum_temp_Q = {channel_out_Q[DWIDTH-1], channel_out_Q} + {final_noise_Q[DWIDTH-1], final_noise_Q};

    // Constantes para saturación (Máximo Positivo y Máximo Negativo)
    localparam signed [DWIDTH-1:0] MAX_POS = {1'b0, {(DWIDTH-1){1'b1}}}; // 011...1
    localparam signed [DWIDTH-1:0] MAX_NEG = {1'b1, {(DWIDTH-1){1'b0}}}; // 100...0

    // Lógica de asignación con saturación
    // Si los dos bits superiores (Signo Extendido y Signo Real) son diferentes, hubo overflow.
    assign Out_I = (sum_temp_I[DWIDTH] ^ sum_temp_I[DWIDTH-1]) ? 
                   (sum_temp_I[DWIDTH] ? MAX_NEG : MAX_POS) : // Hubo overflow -> Saturar
                   sum_temp_I[DWIDTH-1:0];                    // No hubo -> Pasar valor

    assign Out_Q = (sum_temp_Q[DWIDTH] ^ sum_temp_Q[DWIDTH-1]) ? 
                   (sum_temp_Q[DWIDTH] ? MAX_NEG : MAX_POS) : 
                   sum_temp_Q[DWIDTH-1:0];

endmodule