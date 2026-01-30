`timescale 1ns / 1ps

module history_buffer #(
    parameter integer W = 9
)(
    input  wire clk,
    input  wire rst,

    input  wire i_valid,
    input  wire signed [W-1:0] i_X_re,
    input  wire signed [W-1:0] i_X_im,
    input  wire signed [W-1:0] i_W0_re, input  wire signed [W-1:0] i_W0_im,
    input  wire signed [W-1:0] i_W1_re, input  wire signed [W-1:0] i_W1_im,

    output reg  o_valid_data,

    output reg signed [W-1:0] o_X_curr_re, output reg signed [W-1:0] o_X_curr_im,
    output reg signed [W-1:0] o_X_old_re,  output reg signed [W-1:0] o_X_old_im,

    output reg signed [W-1:0] o_W0_re, output reg signed [W-1:0] o_W0_im,
    output reg signed [W-1:0] o_W1_re, output reg signed [W-1:0] o_W1_im,

    output reg [4:0] o_k_idx
);

    // Historial por bin (0..31)
    reg signed [W-1:0] hist_re [0:31];
    reg signed [W-1:0] hist_im [0:31];

    integer ii;
    initial begin
        for (ii = 0; ii < 32; ii = ii + 1) begin
            hist_re[ii] = 0;
            hist_im[ii] = 0;
        end
    end

    reg [4:0] k;

    always @(posedge clk) begin
        if (rst) begin
            k            <= 5'd0;
            o_k_idx      <= 5'd0;
            o_valid_data <= 1'b0;

            o_X_curr_re  <= 0;
            o_X_curr_im  <= 0;
            o_X_old_re   <= 0;
            o_X_old_im   <= 0;

            o_W0_re      <= 0; o_W0_im <= 0;
            o_W1_re      <= 0; o_W1_im <= 0;

        end else begin
            if (i_valid) begin
                // A) índice alineado con las salidas registradas de este ciclo
                o_k_idx <= k;

                // B) leer "viejo" (valor previo guardado)
                o_X_old_re <= hist_re[k];
                o_X_old_im <= hist_im[k];

                // C) actualizar memoria con el "presente"
                hist_re[k] <= i_X_re;
                hist_im[k] <= i_X_im;

                // D) sacar el presente (alineado con old)
                o_X_curr_re <= i_X_re;
                o_X_curr_im <= i_X_im;

                // E) pasar coeficientes (por ahora)
                o_W0_re <= i_W0_re; o_W0_im <= i_W0_im;
                o_W1_re <= i_W1_re; o_W1_im <= i_W1_im;

                // F) valid 1 ciclo después del i_valid (porque es registro)
                o_valid_data <= 1'b1;

                // G) incrementar k (0..31)
                if (k == 5'd31) k <= 5'd0;
                else            k <= k + 5'd1;

            end else begin
                // si no hay datos, reinicio para que el próximo frame arranque k=0
                k            <= 5'd0;
                o_k_idx      <= 5'd0;
                o_valid_data <= 1'b0;
            end
        end
    end

endmodule
