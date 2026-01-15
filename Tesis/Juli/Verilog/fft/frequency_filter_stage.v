`timescale 1ns / 1ps

module frequency_filter_stage #(
    parameter integer W = 16    // Ancho del dato
)(
    input  wire clk,
    input  wire rst,

    // Entradas 
    input  wire i_valid,
    input  wire signed [W-1:0] i_X_re,
    input  wire signed [W-1:0] i_X_im,
    input  wire signed [W-1:0] i_W0_re, input  wire signed [W-1:0] i_W0_im,
    input  wire signed [W-1:0] i_W1_re, input  wire signed [W-1:0] i_W1_im,

  
    output reg o_valid_data, // Avisa al multiplicador que los datos son válidos
    
    output reg signed [W-1:0] o_X_curr_re, output reg signed [W-1:0] o_X_curr_im,
    output reg signed [W-1:0] o_X_old_re,  output reg signed [W-1:0] o_X_old_im,
    
    output reg signed [W-1:0] o_W0_re, output reg signed [W-1:0] o_W0_im,
    output reg signed [W-1:0] o_W1_re, output reg signed [W-1:0] o_W1_im,
    
    // Índice para pedir los W a la memoria externa (igual que antes)
    output reg [4:0] o_k_idx 
);

    // 1. Contador de índice 'k'
    reg [4:0] k;
    always @(posedge clk) begin
        if (rst) k <= 0;
        else if (i_valid) k <= k + 1;
        else k <= 0;
    end
    always @(*) o_k_idx = k;

    // 2. Memoria del Historial (RAM)
    reg signed [W-1:0] hist_re [0:31];
    reg signed [W-1:0] hist_im [0:31];
    
    // Inicialización de memoria a 0 (Para evitar la 'x' roja en simulación)
    integer i;
    initial begin
        for (i=0; i<32; i=i+1) begin
            hist_re[i] = 0;
            hist_im[i] = 0;
        end
    end

    // 3. Pipeline de Alineación y Salida
    // Acá ocurre la magia: Leemos memoria y sacamos todo hacia afuera al mismo tiempo.
    
    always @(posedge clk) begin
        if (i_valid) begin
            // A) SALIDA 1: Recuperamos el Pasado desde la RAM
            // Al asignarlo a una salida (reg), tarda 1 ciclo, igual que leer la RAM.
            o_X_old_re <= hist_re[k];
            o_X_old_im <= hist_im[k];

            // B) ACTUALIZACIÓN: Guardamos el Presente en la RAM (para el futuro)
            hist_re[k] <= i_X_re;
            hist_im[k] <= i_X_im;

            // C) SALIDA 2: Pasamos el Presente hacia afuera
            // Usamos '<=' para crear el retardo de 1 ciclo y que coincida con X_old.
            o_X_curr_re <= i_X_re;
            o_X_curr_im <= i_X_im;
            
            // D) SALIDA 3: Pasamos los Coeficientes hacia afuera (Sincronizados)
            o_W0_re <= i_W0_re; o_W0_im <= i_W0_im;
            o_W1_re <= i_W1_re; o_W1_im <= i_W1_im;
            
            // E) Aviso de validez (También retrasado 1 ciclo)
            o_valid_data <= 1;
            
        end else begin
            o_valid_data <= 0;
            // Opcional: Poner salidas a 0 si querés limpiar ruido, 
            // pero con bajar o_valid_data alcanza.
        end
    end

endmodule