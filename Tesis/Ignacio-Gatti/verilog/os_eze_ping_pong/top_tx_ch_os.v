`timescale 1ns/1ps
`default_nettype none

// ============================================================
// TOP: TX -> Canal (top_ch) -> Overlap-Save (streaming)
//
// clk_fast : clock rápido (FFT / salida OS)
// clk_low  : clock a Fs (stream) generado dividiendo clk_fast por 2
//
// Nota: top_ch incluye ruido via GNG.
//       Para "sin ruido" usando top_ch: sigma_scale = 0.
// ============================================================

module top_tx_ch_os #(
    parameter integer N  = 16,
    parameter integer WN = 9
)(
    input  wire                 clk_fast,
    input  wire                 rst,        // activo alto
    input  wire                 enable_div, // habilita clock_div2

    input  wire [10:0]          sigma_scale,

    output wire                 clk_low,      // debug
    output wire signed [WN-1:0] tx_I_dbg,
    output wire signed [WN-1:0] tx_Q_dbg,
    output wire signed [WN-1:0] ch_I_dbg,
    output wire signed [WN-1:0] ch_Q_dbg,

    output wire                 os_overflow,
    output wire                 os_start,
    output wire                 os_valid,
    output wire signed [WN-1:0] os_I,
    output wire signed [WN-1:0] os_Q
);

    // 1) Clock low = clk_fast/2
    clock_div2 u_div2 (
        .i_clk_fast(clk_fast),
        .i_enable  (enable_div),
        .o_clk_low (clk_low)
    );

    // 2) TX
    reg i_en_tx;
    always @(posedge clk_low) begin
        if (rst) i_en_tx <= 1'b0;
        else     i_en_tx <= 1'b1;
    end

    wire signed [WN-1:0] sI_tx;
    wire signed [WN-1:0] sQ_tx;

    top_tx u_tx (
        .clk   (clk_low),
        .reset (rst),
        .i_en  (i_en_tx),
        .sI_out(sI_tx),
        .sQ_out(sQ_tx)
    );

    assign tx_I_dbg = sI_tx;
    assign tx_Q_dbg = sQ_tx;

    // 3) Canal REAL: top_ch (con gng)
    wire signed [WN-1:0] sI_ch;
    wire signed [WN-1:0] sQ_ch;

    top_ch u_ch (
        .clk        (clk_low),
        .rst        (rst),
        .In_I       (sI_tx),
        .In_Q       (sQ_tx),
        .sigma_scale(sigma_scale),
        .Out_I      (sI_ch),
        .Out_Q      (sQ_ch)
    );

    assign ch_I_dbg = sI_ch;
    assign ch_Q_dbg = sQ_ch;

    // 4) OS streaming
    reg i_valid_low;
    always @(posedge clk_low) begin
        if (rst) i_valid_low <= 1'b0;
        else     i_valid_low <= 1'b1;   // stream continuo
    end

    os_buffer #(
        .N (N),
        .WN(WN)
    ) u_os (
        .i_clk_low (clk_low),
        .i_clk_fast(clk_fast),
        .i_rst     (rst),

        .i_valid   (i_valid_low),
        .i_i       (sI_ch),
        .i_q       (sQ_ch),

        .o_overflow(os_overflow),
        .o_start   (os_start),
        .o_valid   (os_valid),
        .o_i       (os_I),
        .o_q       (os_Q)
    );

endmodule

`default_nettype wire
