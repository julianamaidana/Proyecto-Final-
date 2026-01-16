`timescale 1ns / 1ps

module qpsk_slicer_with_error #(
    parameter integer W    = 16,   // Ancho de palabra
    parameter integer FRAC = 14,   // Bits fraccionarios
    
    
    parameter signed [W-1:0] AMP = 16'sd11585, 
    parameter signed [W-1:0] TH  = 0
)(
    input  wire clk,
    input  wire rst,

    input  wire i_valid,
    input  wire i_first,
    input  wire i_last,
    input  wire signed [W-1:0] i_y_re,
    input  wire signed [W-1:0] i_y_im,

    output reg o_valid,
    output reg o_first,
    output reg o_last,

    // Bits (útiles para contar BER)
    output reg o_bI_hat,
    output reg o_bQ_hat,

    // Símbolo decidido (d)
    output reg signed [W-1:0] o_yhat_re,
    output reg signed [W-1:0] o_yhat_im,

    // --- SALIDA DE ERROR (e = d - y) ---
    output reg signed [W-1:0] o_error_re,
    output reg signed [W-1:0] o_error_im
);

    // Variables temporales para cálculo combinacional dentro del always
    reg signed [W-1:0] dec_re;
    reg signed [W-1:0] dec_im;
    reg negI;
    reg negQ;

    always @(posedge clk) begin
        if (rst) begin
            o_valid    <= 1'b0;
            o_first    <= 1'b0;
            o_last     <= 1'b0;
            o_bI_hat   <= 1'b0;
            o_bQ_hat   <= 1'b0;
            o_yhat_re  <= 0;
            o_yhat_im  <= 0;
            o_error_re <= 0;
            o_error_im <= 0;
        end else begin
            // Valores por defecto (pulso)
            o_valid <= 1'b0;
            o_first <= 1'b0;
            o_last  <= 1'b0;

            if (i_valid) begin
                // 1. Decisión (Lógica combinacional inmediata)
                // Usamos bloqueo (=) para variables temporales
                negI = (i_y_re < TH);
                negQ = (i_y_im < TH);
                
                // 2. Definir valor ideal (d)
                if (negI) dec_re = -AMP; else dec_re = AMP;
                if (negQ) dec_im = -AMP; else dec_im = AMP;

                // 3. Salidas Registradas (<=)
                
                // Bits
                o_bI_hat <= negI;
                o_bQ_hat <= negQ;

                // Símbolo Reconstruido
                o_yhat_re <= dec_re;
                o_yhat_im <= dec_im;

                // CÁLCULO DE ERROR: e = d - y
                o_error_re <= dec_re - i_y_re;
                o_error_im <= dec_im - i_y_im;

                // Control
                o_valid <= 1'b1;
                o_first <= i_first;
                o_last  <= i_last;
            end
        end
    end

endmodule