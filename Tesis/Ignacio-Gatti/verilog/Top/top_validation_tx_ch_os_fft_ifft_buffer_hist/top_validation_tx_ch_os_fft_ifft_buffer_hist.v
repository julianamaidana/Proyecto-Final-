`timescale 1ns / 1ps

module top_validation #(
    parameter DWIDTH      = 9,
    parameter SNR_WIDTH   = 11,
    parameter N_PART      = 16,
    parameter NFFT        = 32
)(
    input  wire                        clk,
    input  wire                        rst,

    input  wire signed [SNR_WIDTH-1:0] sigma_scale,
    input  wire                        bypass_tx,
    input  wire signed [DWIDTH-1:0]    test_data_I,
    input  wire signed [DWIDTH-1:0]    test_data_Q,

    // FFT (como antes)
    output wire                        fft_valid_out,
    output wire signed [8:0]           fft_out_I,
    output wire signed [8:0]           fft_out_Q,

    // NUEVO: salidas del history buffer (para waveform y TB)
    output wire                        hb_valid_out,
    output wire [4:0]                  hb_k_idx,
    output wire signed [8:0]           hb_curr_I,
    output wire signed [8:0]           hb_curr_Q,
    output wire signed [8:0]           hb_old_I,
    output wire signed [8:0]           hb_old_Q
);

    // -----------------------------------------------------------
    // 1) TX+CANAL (tu top original)
    // -----------------------------------------------------------
    wire signed [DWIDTH-1:0] tx_rx_I, tx_rx_Q;

    top #(
        .DWIDTH(DWIDTH),
        .SNR_WIDTH(SNR_WIDTH)
    ) u_system_original (
        .clk         (clk),
        .rst         (rst),
        .sigma_scale (sigma_scale),
        .rx_I        (tx_rx_I),
        .rx_Q        (tx_rx_Q)
    );

    // -----------------------------------------------------------
    // 2) MUX bypass
    // -----------------------------------------------------------
    wire signed [DWIDTH-1:0] buf_in_I = (bypass_tx) ? test_data_I : tx_rx_I;
    wire signed [DWIDTH-1:0] buf_in_Q = (bypass_tx) ? test_data_Q : tx_rx_Q;
    wire                     buf_in_valid = 1'b1;

    // -----------------------------------------------------------
    // 3) Overlap-save buffer (os_buffer)
    // -----------------------------------------------------------
    wire signed [DWIDTH-1:0] os_out_I, os_out_Q;
    wire                     os_valid_w;
    wire                     os_start_w;

    os_buffer #(
        .N (N_PART),
        .WN(DWIDTH)
    ) u_buffer (
        .i_clk       (clk),
        .i_rst       (rst),
        .i_valid     (buf_in_valid),
        .i_xI        (buf_in_I),
        .i_xQ        (buf_in_Q),
        .o_in_ready  (),              // no usado
        .o_fft_start (os_start_w),
        .o_fft_valid (os_valid_w),
        .o_fft_xI    (os_out_I),
        .o_fft_xQ    (os_out_Q)
    );

    // -----------------------------------------------------------
    // 4) FFT
    // -----------------------------------------------------------
    fft_ifft #(
        .NFFT        (NFFT),
        .NB_IN       (DWIDTH),
        .NB_OUT      (9),
        .SCALE_STAGE (0)
    ) u_fft (
        .i_clk      (clk),
        .i_rst      (rst),
        .i_valid    (os_valid_w),
        .i_xI       (os_out_I),
        .i_xQ       (os_out_Q),
        .i_inverse  (1'b0),
        .o_in_ready (),
        .o_start    (),
        .o_valid    (fft_valid_out),
        .o_yI       (fft_out_I),
        .o_yQ       (fft_out_Q)
    );

    // -----------------------------------------------------------
    // 5) HISTORY BUFFER
    // -----------------------------------------------------------
    history_buffer #(
        .W(9)
    ) u_history (
        .clk        (clk),
        .rst        (rst),

        .i_valid    (fft_valid_out),
        .i_X_re     (fft_out_I),
        .i_X_im     (fft_out_Q),

        // pesos dummy (por ahora)
        .i_W0_re    (9'sd0), .i_W0_im(9'sd0),
        .i_W1_re    (9'sd0), .i_W1_im(9'sd0),

        .o_valid_data (hb_valid_out),
        .o_k_idx      (hb_k_idx),

        .o_X_curr_re  (hb_curr_I),
        .o_X_curr_im  (hb_curr_Q),
        .o_X_old_re   (hb_old_I),
        .o_X_old_im   (hb_old_Q),

        .o_W0_re(), .o_W0_im(),
        .o_W1_re(), .o_W1_im()
    );

endmodule
