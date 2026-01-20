`timescale 1ns/1ps

module top_tx_ch_os_fft_ifft #(
    parameter integer DWIDTH      = 9,
    parameter integer DATA_F      = 7,
    parameter integer SNR_WIDTH   = 11,
    parameter integer OS_N        = 16,

    parameter integer NFFT        = 32,
    parameter integer LOGN        = 5,
    parameter integer NB_W        = 17,
    parameter integer NBF_W       = 10
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire signed [SNR_WIDTH-1:0] sigma_scale,

    output wire signed [DWIDTH-1:0]    rx_I,
    output wire signed [DWIDTH-1:0]    rx_Q,

    output wire                        os_in_ready,
    output wire                        os_fft_start,
    output wire                        os_fft_valid,
    output wire signed [DWIDTH-1:0]    os_fft_xI,
    output wire signed [DWIDTH-1:0]    os_fft_xQ,

    output wire                        fft_in_ready,
    output wire                        fft_start,
    output wire                        fft_valid,
    output wire signed [NB_W-1:0]      fft_yI_w,
    output wire signed [NB_W-1:0]      fft_yQ_w,

    output wire                        ifft_in_ready,
    output wire                        ifft_start,
    output wire                        ifft_valid,
    output wire signed [DWIDTH-1:0]    ifft_yI,
    output wire signed [DWIDTH-1:0]    ifft_yQ
);

    // =========================
    // TX
    // =========================
    wire signed [15:0] tx_I_full;
    wire signed [15:0] tx_Q_full;

    wire signed [DWIDTH-1:0] tx_I_internal = tx_I_full[DWIDTH-1:0];
    wire signed [DWIDTH-1:0] tx_Q_internal = tx_Q_full[DWIDTH-1:0];

    tx_top u_tx_top (
        .clk    (clk),
        .reset  (rst),
        .sI_out (tx_I_full),
        .sQ_out (tx_Q_full)
    );

    // =========================
    // Canal + ruido
    // =========================
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

    assign rx_I = chan_I;
    assign rx_Q = chan_Q;

    // =========================
    // Overlap-Save
    // =========================
    wire os_ready_internal;

    // Alimentamos OS sólo si:
    //  - OS está colectando
    //  - FFT está lista para colectar
    wire os_i_valid = os_ready_internal & fft_in_ready;

    os_buffer #(
        .N  (OS_N),
        .WN (DWIDTH)
    ) u_os_buffer (
        .i_clk      (clk),
        .i_rst      (rst),
        .i_valid    (os_i_valid),
        .i_xI       (chan_I),
        .i_xQ       (chan_Q),

        .o_in_ready (os_ready_internal),

        .o_fft_start(os_fft_start),
        .o_fft_valid(os_fft_valid),
        .o_fft_xI   (os_fft_xI),
        .o_fft_xQ   (os_fft_xQ)
    );

    assign os_in_ready = os_ready_internal;

    // =========================
    // FFT (narrow -> wide)
    // =========================
    fft_ifft #(
        .NFFT(NFFT),
        .LOGN(LOGN),
        .NB_IN(DWIDTH),
        .NBF_IN(DATA_F),
        .NB_W(NB_W),
        .NBF_W(NBF_W),
        .NB_OUT(NB_W),
        .NBF_OUT(NBF_W),
        .SCALE_STAGE(1),
        .FINAL_INV_SCALE(0)
    ) u_fft (
        .i_clk     (clk),
        .i_rst     (rst),
        .i_valid   (os_fft_valid),
        .i_xI      (os_fft_xI),
        .i_xQ      (os_fft_xQ),
        .i_inverse (1'b0),

        .o_in_ready(fft_in_ready),
        .o_start   (fft_start),
        .o_valid   (fft_valid),
        .o_yI      (fft_yI_w),
        .o_yQ      (fft_yQ_w)
    );

    // =========================
    // IFFT (wide -> narrow)
    // =========================
    fft_ifft #(
        .NFFT(NFFT),
        .LOGN(LOGN),
        .NB_IN(NB_W),
        .NBF_IN(NBF_W),
        .NB_W(NB_W),
        .NBF_W(NBF_W),
        .NB_OUT(DWIDTH),
        .NBF_OUT(DATA_F),
        .SCALE_STAGE(0),
        .FINAL_INV_SCALE(0)
    ) u_ifft (
        .i_clk     (clk),
        .i_rst     (rst),
        .i_valid   (fft_valid),
        .i_xI      (fft_yI_w),
        .i_xQ      (fft_yQ_w),
        .i_inverse (1'b1),

        .o_in_ready(ifft_in_ready),
        .o_start   (ifft_start),
        .o_valid   (ifft_valid),
        .o_yI      (ifft_yI),
        .o_yQ      (ifft_yQ)
    );

endmodule
