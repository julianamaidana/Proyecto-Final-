`timescale 1ns / 1ps

module time_weight_update #(
    parameter integer W      = 16,   // Ancho de dato
    parameter integer N_FFT  = 32,   // Tamaño del bloque que entra/sale (Total FFT)
    parameter integer N_TAPS = 16,   // Cantidad de coeficientes reales (Memoria)
    parameter integer MU_SHIFT = 6,  // Paso de adaptación
    
    // Configuración del Impulso Inicial
    parameter integer CENTER_IDX = 15,    // Posición del "1"
    parameter integer ONE_FIXED  = 1024   // Valor de "1.0" en punto fijo (ej. Q.10)
)(
    input  wire clk,
    input  wire rst,

    // Entrada: Viene de Projection (Paquetes de 32 muestras)
    input  wire i_valid,
    input  wire i_last, 
    input  wire signed [W-1:0] i_grad_re, // Trae 16 gradientes + 16 ceros
    input  wire signed [W-1:0] i_grad_im,

    // Salida: Va a la FFT de Pesos (Paquetes de 32 muestras)
    output reg  o_valid,
    output reg  o_last,
    output reg  signed [W-1:0] o_weight_re,
    output reg  signed [W-1:0] o_weight_im
);

    // --- 1. Memoria de Coeficientes (Solo N_TAPS de profundidad) ---
    // No gastamos memoria en la parte de padding
    reg signed [W-1:0] mem_w_re [0:N_TAPS-1];
    reg signed [W-1:0] mem_w_im [0:N_TAPS-1];

    // --- 2. Control ---
    // Contador para saber si estamos en la parte baja (0-15) o alta (16-31)
    reg [$clog2(N_FFT):0] cnt;
    integer k;

    // --- 3. Cálculo de Actualización ---
    // Solo es válido si cnt < N_TAPS
    wire signed [W-1:0] w_old_re;
    wire signed [W-1:0] w_old_im;
    
    // Leemos memoria (si cnt se pasa de 15, lee basura pero no la usamos)
    assign w_old_re = (cnt < N_TAPS) ? mem_w_re[cnt] : {W{1'b0}};
    assign w_old_im = (cnt < N_TAPS) ? mem_w_im[cnt] : {W{1'b0}};

    // Suma: w_new = w_old + (grad >>> mu)
    wire signed [W-1:0] w_new_re = w_old_re + (i_grad_re >>> MU_SHIFT);
    wire signed [W-1:0] w_new_im = w_old_im + (i_grad_im >>> MU_SHIFT);

    // --- 4. Proceso Principal ---
    always @(posedge clk) begin
        if (rst) begin
            o_valid     <= 0;
            o_last      <= 0;
            o_weight_re <= 0;
            o_weight_im <= 0;
            cnt         <= 0;

            // --- INICIALIZACIÓN (Reset) ---
            for (k=0; k < N_TAPS; k=k+1) begin
                if (k == CENTER_IDX) 
                    mem_w_re[k] <= ONE_FIXED; // Impulso
                else                 
                    mem_w_re[k] <= 0;         // Ceros
                
                mem_w_im[k] <= 0;
            end

        end else begin
            
            o_valid <= 0;
            o_last  <= 0;

            if (i_valid) begin
                o_valid <= 1;

                // --- CASO A: Parte Baja (0 a 15) ---
                // Hay gradiente útil -> Actualizamos Memoria
                if (cnt < N_TAPS) begin
                    // 1. Guardar en RAM
                    mem_w_re[cnt] <= w_new_re;
                    mem_w_im[cnt] <= w_new_im;
                    
                    // 2. Sacar a la salida
                    o_weight_re   <= w_new_re;
                    o_weight_im   <= w_new_im;
                end 
                
                // --- CASO B: Parte Alta (16 a 31) ---
                // Es zona de Padding -> Mandamos Ceros a la FFT
                else begin
                    // No tocamos la RAM
                    o_weight_re   <= 0;
                    o_weight_im   <= 0;
                end

                // --- Control de LAST y Contador ---
                if (i_last || cnt == N_FFT-1) begin
                    o_last <= 1;
                    cnt    <= 0;
                end else begin
                    cnt <= cnt + 1;
                end
            end
        end
    end

endmodule