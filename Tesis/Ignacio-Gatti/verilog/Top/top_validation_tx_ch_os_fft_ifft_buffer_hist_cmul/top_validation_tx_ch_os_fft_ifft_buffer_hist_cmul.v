`timescale 1ns/1ps

module top_validation #(
    parameter integer DWIDTH    = 9,
    parameter integer SNR_WIDTH = 11,
    parameter integer N_PART    = 16,
    parameter integer NFFT      = 32
)(
    input  wire                        clk,
    input  wire                        rst,

    input  wire signed [SNR_WIDTH-1:0] sigma_scale,
    input  wire                        bypass_tx,
    input  wire signed [DWIDTH-1:0]    test_data_I,
    input  wire signed [DWIDTH-1:0]    test_data_Q,

    // Coeficientes (Q7 en 9 bits)
    input  wire signed [DWIDTH-1:0]    i_W0_re,
    input  wire signed [DWIDTH-1:0]    i_W0_im,
    input  wire signed [DWIDTH-1:0]    i_W1_re,
    input  wire signed [DWIDTH-1:0]    i_W1_im,

    // FFT (para mirar)
    output wire                        fft_valid_out,
    output wire signed [DWIDTH-1:0]    fft_out_I,
    output wire signed [DWIDTH-1:0]    fft_out_Q,

    // HISTORY (para mirar)
    output wire                        hb_valid_out,
    output wire [4:0]                  hb_k_idx,
    output wire signed [DWIDTH-1:0]    hb_curr_I,
    output wire signed [DWIDTH-1:0]    hb_curr_Q,
    output wire signed [DWIDTH-1:0]    hb_old_I,
    output wire signed [DWIDTH-1:0]    hb_old_Q,

    // SALIDA CMUL+SUM (para mirar)
    output wire                        y_valid,
    output wire [4:0]                  y_k_idx,
    output wire signed [DWIDTH-1:0]    y_I,
    output wire signed [DWIDTH-1:0]    y_Q
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
        .o_in_ready  (),
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
    // 5) HISTORY BUFFER (te entrega Xcurr y Xold alineados por k)
    // -----------------------------------------------------------
    history_buffer #(
        .W(9)
    ) u_history (
        .clk          (clk),
        .rst          (rst),

        .i_valid      (fft_valid_out),
        .i_X_re       (fft_out_I),
        .i_X_im       (fft_out_Q),

        .i_W0_re      (i_W0_re),
        .i_W0_im      (i_W0_im),
        .i_W1_re      (i_W1_re),
        .i_W1_im      (i_W1_im),

        .o_valid_data (hb_valid_out),
        .o_k_idx      (hb_k_idx),

        .o_X_curr_re  (hb_curr_I),
        .o_X_curr_im  (hb_curr_Q),
        .o_X_old_re   (hb_old_I),
        .o_X_old_im   (hb_old_Q),

        .o_W0_re(), .o_W0_im(),
        .o_W1_re(), .o_W1_im()
    );

    // -----------------------------------------------------------
    // 6) CMUL: Y = Xcurr*W0 + Xold*W1  (en frecuencia, por bin k)
    // -----------------------------------------------------------

    // (a) multiplicaciones complejas
    wire signed [DWIDTH-1:0] y0_I, y0_Q;
    wire signed [DWIDTH-1:0] y1_I, y1_Q;

    complex_mult #(
        .NB_W (9),
        .NBF_W(7)
    ) u_cmul0 (
        .i_aI (hb_curr_I),
        .i_aQ (hb_curr_Q),
        .i_bI (i_W0_re),
        .i_bQ (i_W0_im),
        .o_yI (y0_I),
        .o_yQ (y0_Q)
    );

    complex_mult #(
        .NB_W (9),
        .NBF_W(7)
    ) u_cmul1 (
        .i_aI (hb_old_I),
        .i_aQ (hb_old_Q),
        .i_bI (i_W1_re),
        .i_bQ (i_W1_im),
        .o_yI (y1_I),
        .o_yQ (y1_Q)
    );

    // (b) suma saturada a 9b signed [-256..255]
    function signed [8:0] sat9;
        input signed [31:0] x;
        begin
            if (x > 255)       sat9 = 9'sd255;
            else if (x < -256) sat9 = -9'sd256;
            else               sat9 = x[8:0];
        end
    endfunction

    wire signed [31:0] sumI_full = $signed(y0_I) + $signed(y1_I);
    wire signed [31:0] sumQ_full = $signed(y0_Q) + $signed(y1_Q);

    assign y_I = sat9(sumI_full);
    assign y_Q = sat9(sumQ_full);

    // valids e índice (combinacional, mismo ciclo que hb_valid_out)
    assign y_valid = hb_valid_out;
    assign y_k_idx = hb_k_idx;

endmodule
