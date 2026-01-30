`timescale 1ns/1ps

module cmul_2part #(
    parameter integer NB   = 9,
    parameter integer NBF  = 7
)(
    input  wire                 i_clk,
    input  wire                 i_rst,

    input  wire                 i_valid,
    input  wire [4:0]           i_k,

    input  wire signed [NB-1:0] i_xc_re,
    input  wire signed [NB-1:0] i_xc_im,
    input  wire signed [NB-1:0] i_xo_re,
    input  wire signed [NB-1:0] i_xo_im,

    input  wire signed [NB-1:0] i_w0_re,
    input  wire signed [NB-1:0] i_w0_im,
    input  wire signed [NB-1:0] i_w1_re,
    input  wire signed [NB-1:0] i_w1_im,

    output reg                  o_valid,
    output reg  [4:0]           o_k,
    output reg  signed [NB-1:0] o_y_re,
    output reg  signed [NB-1:0] o_y_im
);

    // Productos complejos
    wire signed [NB-1:0] p0_re, p0_im;
    wire signed [NB-1:0] p1_re, p1_im;

    complex_mult #(.NB_W(NB), .NBF_W(NBF)) u_mul0 (
        .i_a_re(i_xc_re), .i_a_im(i_xc_im),
        .i_b_re(i_w0_re), .i_b_im(i_w0_im),
        .o_y_re(p0_re),   .o_y_im(p0_im)
    );

    complex_mult #(.NB_W(NB), .NBF_W(NBF)) u_mul1 (
        .i_a_re(i_xo_re), .i_a_im(i_xo_im),
        .i_b_re(i_w1_re), .i_b_im(i_w1_im),
        .o_y_re(p1_re),   .o_y_im(p1_im)
    );

    // Suma con 1 bit extra
    wire signed [NB:0] sum_re = $signed({p0_re[NB-1], p0_re}) + $signed({p1_re[NB-1], p1_re});
    wire signed [NB:0] sum_im = $signed({p0_im[NB-1], p0_im}) + $signed({p1_im[NB-1], p1_im});

    // Saturación a NB bits (misma fracción)
    wire signed [NB-1:0] y_re_sat;
    wire signed [NB-1:0] y_im_sat;

    sat_trunc #(
        .NB_XI(NB+1), .NBF_XI(NBF),
        .NB_XO(NB),   .NBF_XO(NBF),
        .ROUND_EVEN(0)
    ) u_sat_re (
        .i_data(sum_re),
        .o_data(y_re_sat)
    );

    sat_trunc #(
        .NB_XI(NB+1), .NBF_XI(NBF),
        .NB_XO(NB),   .NBF_XO(NBF),
        .ROUND_EVEN(0)
    ) u_sat_im (
        .i_data(sum_im),
        .o_data(y_im_sat)
    );

    // Registro de salida alineado con valid/k
    always @(posedge i_clk) begin
        if (i_rst) begin
            o_valid <= 1'b0;
            o_k     <= 5'd0;
            o_y_re  <= {NB{1'b0}};
            o_y_im  <= {NB{1'b0}};
        end else begin
            o_valid <= i_valid;
            o_k     <= i_k;
            o_y_re  <= y_re_sat;
            o_y_im  <= y_im_sat;
        end
    end

endmodule
