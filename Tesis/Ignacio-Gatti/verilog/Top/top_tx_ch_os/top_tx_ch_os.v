`timescale 1ns/1ps

module top_tx_ch_os #(
    parameter DWIDTH      = 9,   // coincide con os_buffer WN y con canal
    parameter SNR_WIDTH   = 11,
    parameter OS_N        = 16    // N del overlap-save => emite 2N
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire signed [SNR_WIDTH-1:0] sigma_scale,

    // Salida del canal (muestra a muestra)
    output wire signed [DWIDTH-1:0]    rx_I,
    output wire signed [DWIDTH-1:0]    rx_Q,

    // Salida del Overlap-Save hacia tu FFT (stream de 2N)
    output wire                        os_in_ready,
    output wire                        os_fft_start,
    output wire                        os_fft_valid,
    output wire signed [DWIDTH-1:0]    os_fft_xI,
    output wire signed [DWIDTH-1:0]    os_fft_xQ
);

    // ============================================================
    // 1) TX
    // ============================================================
    wire signed [15:0] tx_I_full;
    wire signed [15:0] tx_Q_full;

    // versión "narrow" que usás en el resto del sistema
    wire signed [DWIDTH-1:0] tx_I_internal = tx_I_full[DWIDTH-1:0];
    wire signed [DWIDTH-1:0] tx_Q_internal = tx_Q_full[DWIDTH-1:0];

    tx_top u_tx_top (
        .clk    (clk),
        .reset  (rst),
        .sI_out (tx_I_full),
        .sQ_out (tx_Q_full)
    );

    // ============================================================
    // 2) CANAL + RUIDO
    // ============================================================
    wire signed [DWIDTH-1:0] chan_I;
    wire signed [DWIDTH-1:0] chan_Q;

    channel_with_noise #(
        .DWIDTH(DWIDTH),
        .SNR_WIDTH(SNR_WIDTH)
    ) u_channel (
        .clk         (clk),
        .rst         (rst),
        .In_I        (tx_I_internal),
        .In_Q        (tx_Q_internal),
        .sigma_scale (sigma_scale),
        .Out_I       (chan_I),
        .Out_Q       (chan_Q)
    );

    // Exponemos también la salida del canal (como antes)
    assign rx_I = chan_I;
    assign rx_Q = chan_Q;

    // ============================================================
    // 3) OVERLAP-SAVE BUFFER
    // ============================================================
    // Alimentamos el buffer sólo cuando está en COLLECT (ready=1)
    // Para esta etapa de integración/test, esto valida bien el bloque.
    wire os_i_valid = os_in_ready;

    os_buffer #(
        .N  (OS_N),
        .WN (DWIDTH)
    ) u_os_buffer (
        .i_clk      (clk),
        .i_rst      (rst),
        .i_valid    (os_i_valid),
        .i_xI       (chan_I),
        .i_xQ       (chan_Q),

        .o_in_ready (os_in_ready),

        .o_fft_start(os_fft_start),
        .o_fft_valid(os_fft_valid),
        .o_fft_xI   (os_fft_xI),
        .o_fft_xQ   (os_fft_xQ)
    );

endmodule
