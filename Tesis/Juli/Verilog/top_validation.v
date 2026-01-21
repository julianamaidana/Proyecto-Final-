`timescale 1ns / 1ps

module top_validation #(
    parameter DWIDTH      = 9,
    parameter SNR_WIDTH   = 11,
    parameter N_PART      = 16, // Tamaño de partición (N)
    parameter NFFT        = 32  // Tamaño FFT (2*N)
)(
    input  wire                        clk,
    input  wire                        rst,
    
    // Control de Ruido y Mux
    input  wire signed [SNR_WIDTH-1:0] sigma_scale, 
    input  wire                        bypass_tx,   // 1=Usar datos prueba, 0=Usar Tx Real
    input  wire signed [DWIDTH-1:0]    test_data_I, // Dato manual (para probar bines)
    input  wire signed [DWIDTH-1:0]    test_data_Q,

    // Salidas para ver en el Waveform
    output wire                        fft_valid_out,
    output wire signed [8:0]           fft_out_I,   
    output wire signed [8:0]           fft_out_Q
);

    // -----------------------------------------------------------
    // 1. CABLES INTERNOS
    // -----------------------------------------------------------
    // Salidas del Top original
    wire signed [DWIDTH-1:0] tx_rx_I, tx_rx_Q;

    // Entradas al Buffer (Salida del MUX)
    wire signed [DWIDTH-1:0] buf_in_I, buf_in_Q;
    wire                     buf_in_valid;

    // Salidas del Buffer -> Entradas FFT
    wire signed [DWIDTH-1:0] os_out_I, os_out_Q;
    wire                     os_valid_w;
    wire                     os_start_w;

    // -----------------------------------------------------------
    // 2. INSTANCIA DE TU SISTEMA (TX + CANAL)
    // -----------------------------------------------------------
    top #(
        .DWIDTH(DWIDTH), 
        .SNR_WIDTH(SNR_WIDTH)
    ) u_system_original (
        .clk        (clk),
        .rst        (rst),
        .sigma_scale(sigma_scale), // Control de ruido
        .rx_I       (tx_rx_I),     // Conectamos la salida
        .rx_Q       (tx_rx_Q)
    );

    // -----------------------------------------------------------
    // 3. MULTIPLEXOR DE PRUEBA (BYPASS)
    // -----------------------------------------------------------
    // Si bypass_tx=1, entra el dato manual (ej. constante 100).
    // Si bypass_tx=0, entra lo que sale de u_system_original.
    assign buf_in_I = (bypass_tx) ? test_data_I : tx_rx_I;
    assign buf_in_Q = (bypass_tx) ? test_data_Q : tx_rx_Q;
    assign buf_in_valid = 1'b1; // Asumimos flujo continuo

    // -----------------------------------------------------------
    // 4. INSTANCIA DEL BUFFER OVERLAP-SAVE
    // -----------------------------------------------------------
    os_buffer #(
        .N (N_PART), 
        .WN(DWIDTH)
    ) u_buffer (
        .i_clk      (clk),
        .i_rst      (rst),
        .i_valid    (buf_in_valid),
        .i_xI       (buf_in_I),
        .i_xQ       (buf_in_Q),
        
        .o_in_ready (),           // No la usamos en flujo continuo
        .o_fft_start(os_start_w), // Pulso de inicio
        .o_fft_valid(os_valid_w), // Ventana de validez (2N ciclos)
        .o_fft_xI   (os_out_I),   // Datos ordenados para FFT
        .o_fft_xQ   (os_out_Q)
    );

    // -----------------------------------------------------------
    // 5. INSTANCIA DE TU FFT
    // -----------------------------------------------------------
    fft_ifft #(
        .NFFT       (NFFT),
        .NB_IN      (DWIDTH),
        .NB_OUT     (9),
        .SCALE_STAGE(0) // Sin escalar para ver picos grandes
    ) u_fft (
        .i_clk      (clk),
        .i_rst      (rst),
        
        // Conexión con el Buffer
        .i_valid    (os_valid_w), 
        .i_xI       (os_out_I),
        .i_xQ       (os_out_Q),
        .i_inverse  (1'b0),       // 0 = FFT Directa (Forward)

        // Salidas
        .o_in_ready (),           
        .o_start    (),           
        .o_valid    (fft_valid_out),
        .o_yI       (fft_out_I),
        .o_yQ       (fft_out_Q)
    );

    // ... (Después de la instancia u_fft) ...

    // CABLES PARA EL BUFFER DE HISTORIA
    wire hb_valid_out;
    wire signed [8:0] hb_curr_I, hb_curr_Q;
    wire signed [8:0] hb_old_I, hb_old_Q;
    wire [4:0] hb_k_idx; // Índice k (0 a 31)

    // INSTANCIA DEL HISTORY BUFFER
    history_buffer #(
        .W(9) // Ancho de 9 bits
    ) u_history_buf (
        .clk(clk),
        .rst(rst),
        .i_valid(fft_valid_out), // Conectamos el VALID de la FFT
        
        // Entradas de Datos (Desde la FFT)
        .i_X_re(fft_out_I),
        .i_X_im(fft_out_Q),
        
        // Entradas de Pesos (Por ahora pon ceros o conecta cables dummy)
        .i_W0_re(9'sd0), .i_W0_im(9'sd0),
        .i_W1_re(9'sd0), .i_W1_im(9'sd0),

        // Salidas
        .o_valid_data(hb_valid_out), // Nuevo Valid retrasado
        .o_X_curr_re(hb_curr_I), .o_X_curr_im(hb_curr_Q), // Dato Actual
        .o_X_old_re (hb_old_I),  .o_X_old_im (hb_old_Q),  // Dato Viejo
        
        .o_k_idx(hb_k_idx),
        // Salidas de pesos (No las miramos por ahora)
        .o_W0_re(), .o_W0_im(), .o_W1_re(), .o_W1_im()
    );
endmodule