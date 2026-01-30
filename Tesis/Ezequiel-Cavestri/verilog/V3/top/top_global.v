module top_global #(
    parameter integer DWIDTH      = 9,
    parameter integer SNR_WIDTH   = 11,
    parameter integer N_PART      = 16,
    parameter integer NFFT        = 32,
    parameter integer IDEAL_CH    = 1,

    parameter integer NB_WIDE     = 22,
    parameter integer NBF_WIDE    = 10
)(
    input  wire                        clk,
    input  wire                        rst,

    input  wire signed [SNR_WIDTH-1:0] sigma_scale,

    input  wire                        bypass_tx,
    input  wire signed [DWIDTH-1:0]    test_data_I,
    input  wire signed [DWIDTH-1:0]    test_data_Q,

    output wire signed [DWIDTH-1:0]    tx_sym_I,
    output wire signed [DWIDTH-1:0]    tx_sym_Q,
    output wire signed [DWIDTH-1:0]    ch_out_I,
    output wire signed [DWIDTH-1:0]    ch_out_Q,

    // FFT 9b (debug/history)
    output wire                        fft_valid_out,
    output wire signed [8:0]           fft_out_I,
    output wire signed [8:0]           fft_out_Q,

    // IFFT 9b (debug)
    output wire                        ifft_valid_out,
    output wire signed [8:0]           ifft_out_I,
    output wire signed [8:0]           ifft_out_Q,

    // WIDE taps for TB VM
    output wire                        fft_w_valid_out,
    output wire signed [NB_WIDE-1:0]   fft_w_out_I,
    output wire signed [NB_WIDE-1:0]   fft_w_out_Q,
    output wire                        ifft_w_valid_out,
    output wire signed [NB_WIDE-1:0]   ifft_w_out_I,
    output wire signed [NB_WIDE-1:0]   ifft_w_out_Q,

    // HISTORY
    output wire                        hb_valid_out,
    output wire [4:0]                  hb_k_idx,
    output wire signed [8:0]           hb_curr_I,
    output wire signed [8:0]           hb_curr_Q,
    output wire signed [8:0]           hb_old_I,
    output wire signed [8:0]           hb_old_Q
);

    // ============================================================
    // os_buffer ready
    // ============================================================
    wire os_in_ready_w;

    // ============================================================
    // FFT/IFFT readiness (para no largar un frame si se va a perder)
    // ============================================================
    wire fft_in_ready_w;
    wire ifft_in_ready_w;

    // ============================================================
    // Contador de samples "nuevos" dentro del bloque de N_PART
    // ============================================================
    reg [$clog2(N_PART)-1:0] blk_cnt;
    wire last_sample = (blk_cnt == N_PART-1);

    // ============================================================
    // DRIVE VALID: SOLO consumimos un sample cuando:
    // - os esta listo para colectar (os_in_ready_w)
    // - si es el ultimo sample, garantizamos fft_in_ready_w (como tu TB)
    // - ademas, no arrancamos un frame si la IFFT no esta lista para colectar
    //   (evita que FFT mande datos cuando IFFT esta en SEND/ocupada)
    // ============================================================
    wire drive_valid = os_in_ready_w
                     & (~last_sample | fft_in_ready_w)
                     & ifft_in_ready_w;

    // ============================================================
    // TX PRBS: avanza solo cuando realmente consumimos un sample
    // ============================================================
    wire tx_en_w = drive_valid & ~bypass_tx;

    wire signed [DWIDTH-1:0] tx_I, tx_Q;

    top_tx u_tx (
        .clk   (clk),
        .reset (rst),
        .i_en  (tx_en_w),
        .sI_out(tx_I),
        .sQ_out(tx_Q)
    );

    assign tx_sym_I = tx_I;
    assign tx_sym_Q = tx_Q;

    // ============================================================
    // Sample actual a inyectar al canal (combinacional)
    // (esto es lo que el OS debe ver cuando drive_valid=1)
    // ============================================================
    wire signed [DWIDTH-1:0] drive_I = (bypass_tx) ? test_data_I : tx_sym_I;
    wire signed [DWIDTH-1:0] drive_Q = (bypass_tx) ? test_data_Q : tx_sym_Q;

    // ============================================================
    // Registros de debug/estabilidad (solo cambian cuando drive_valid=1)
    // ============================================================
    reg signed [DWIDTH-1:0] src_I, src_Q;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            src_I   <= 'sd0;
            src_Q   <= 'sd0;
            blk_cnt <= 'd0;
        end else begin
            if (drive_valid) begin
                src_I <= drive_I;
                src_Q <= drive_Q;

                if (last_sample) blk_cnt <= 'd0;
                else             blk_cnt <= blk_cnt + 1'b1;
            end
        end
    end

    // ============================================================
    // Canal real o ideal
    // - Ideal: SIN latencia (combinacional)
    // - Real : usa top_ch
    // ============================================================
    wire signed [DWIDTH-1:0] ch_real_I, ch_real_Q;

    top_ch #(
        .DWIDTH    (DWIDTH),
        .SNR_WIDTH (SNR_WIDTH)
    ) u_ch (
        .clk         (clk),
        .rst         (rst),
        .In_I        (src_I),
        .In_Q        (src_Q),
        .sigma_scale (sigma_scale),
        .Out_I       (ch_real_I),
        .Out_Q       (ch_real_Q)
    );

    generate
        if (IDEAL_CH) begin : g_ideal
            // ideal sin latencia
            assign ch_out_I = src_I;
            assign ch_out_Q = src_Q;
        end else begin : g_real
            assign ch_out_I = ch_real_I;
            assign ch_out_Q = ch_real_Q;
        end
    endgenerate

    // ============================================================
    // os_buffer
    // IMPORTANTISIMO: i_valid = drive_valid (no constante 1)
    // ============================================================
    wire signed [DWIDTH-1:0] rx_in_I = ch_out_I;
    wire signed [DWIDTH-1:0] rx_in_Q = ch_out_Q;
    wire                     rx_in_valid = drive_valid;

    wire signed [DWIDTH-1:0] os_out_I, os_out_Q;
    wire                     os_valid_w;
    wire                     os_start_w;

    os_buffer #(
        .N  (N_PART),
        .WN (DWIDTH)
    ) u_os (
        .i_clk       (clk),
        .i_rst       (rst),
        .i_valid     (rx_in_valid),
        .i_xI        (rx_in_I),
        .i_xQ        (rx_in_Q),
        .o_in_ready  (os_in_ready_w),
        .o_fft_start (os_start_w),
        .o_fft_valid (os_valid_w),
        .o_fft_xI    (os_out_I),
        .o_fft_xQ    (os_out_Q)
    );

    // ============================================================
    // FFT WIDE
    // ============================================================
    wire                     fft_start_w;
    wire                     fft_w_valid;
    wire signed [NB_WIDE-1:0] fft_w_I;
    wire signed [NB_WIDE-1:0] fft_w_Q;

    fft_ifft #(
        .NFFT        (NFFT),
        .LOGN        (5),
        .NB_IN       (DWIDTH),
        .NBF_IN      (7),
        .NB_W        (NB_WIDE),
        .NBF_W       (NBF_WIDE),
        .NB_OUT      (NB_WIDE),
        .NBF_OUT     (NBF_WIDE),
        .SCALE_STAGE (0)
    ) u_fft_wide (
        .i_clk      (clk),
        .i_rst      (rst),
        .i_valid    (os_valid_w),
        .i_xI       (os_out_I),
        .i_xQ       (os_out_Q),
        .i_inverse  (1'b0),
        .o_in_ready (fft_in_ready_w),
        .o_start    (fft_start_w),
        .o_valid    (fft_w_valid),
        .o_yI       (fft_w_I),
        .o_yQ       (fft_w_Q)
    );

    assign fft_w_valid_out = fft_w_valid;
    assign fft_w_out_I     = fft_w_I;
    assign fft_w_out_Q     = fft_w_Q;

    // ============================================================
    // FFT narrow 9b (debug/history)
    // ============================================================
    wire signed [8:0] fft9_I;
    wire signed [8:0] fft9_Q;

    sat_trunc #(
        .NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE),
        .NB_XO(9),       .NBF_XO(7),
        .ROUND_EVEN(1)
    ) u_fft9_I (
        .i_data(fft_w_I),
        .o_data(fft9_I)
    );

    sat_trunc #(
        .NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE),
        .NB_XO(9),       .NBF_XO(7),
        .ROUND_EVEN(1)
    ) u_fft9_Q (
        .i_data(fft_w_Q),
        .o_data(fft9_Q)
    );

    assign fft_valid_out = fft_w_valid;
    assign fft_out_I     = fft9_I;
    assign fft_out_Q     = fft9_Q;

    // ============================================================
    // IFFT WIDE
    // ============================================================
    wire                      ifft_start_w;
    wire                      ifft_w_valid;
    wire signed [NB_WIDE-1:0] ifft_w_I;
    wire signed [NB_WIDE-1:0] ifft_w_Q;

    fft_ifft #(
        .NFFT        (NFFT),
        .LOGN        (5),
        .NB_IN       (NB_WIDE),
        .NBF_IN      (NBF_WIDE),
        .NB_W        (NB_WIDE),
        .NBF_W       (NBF_WIDE),
        .NB_OUT      (NB_WIDE),
        .NBF_OUT     (NBF_WIDE),
        .SCALE_STAGE (0)
    ) u_ifft_wide (
        .i_clk      (clk),
        .i_rst      (rst),
        .i_valid    (fft_w_valid),
        .i_xI       (fft_w_I),
        .i_xQ       (fft_w_Q),
        .i_inverse  (1'b1),
        .o_in_ready (ifft_in_ready_w),
        .o_start    (ifft_start_w),
        .o_valid    (ifft_w_valid),
        .o_yI       (ifft_w_I),
        .o_yQ       (ifft_w_Q)
    );

    assign ifft_w_valid_out = ifft_w_valid;
    assign ifft_w_out_I     = ifft_w_I;
    assign ifft_w_out_Q     = ifft_w_Q;

    // ============================================================
    // IFFT narrow 9b (debug)
    // ============================================================
    sat_trunc #(
        .NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE),
        .NB_XO(9),       .NBF_XO(7),
        .ROUND_EVEN(1)
    ) u_ifft9_I (
        .i_data(ifft_w_I),
        .o_data(ifft_out_I)
    );

    sat_trunc #(
        .NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE),
        .NB_XO(9),       .NBF_XO(7),
        .ROUND_EVEN(1)
    ) u_ifft9_Q (
        .i_data(ifft_w_Q),
        .o_data(ifft_out_Q)
    );

    assign ifft_valid_out = ifft_w_valid;

    // ============================================================
    // History buffer (usa FFT 9b)
    // ============================================================
    history_buffer #(
        .W(9)
    ) u_history (
        .clk          (clk),
        .rst          (rst),
        .i_valid      (fft_valid_out),
        .i_X_re       (fft_out_I),
        .i_X_im       (fft_out_Q),
        .i_W0_re      (9'sd0), .i_W0_im(9'sd0),
        .i_W1_re      (9'sd0), .i_W1_im(9'sd0),
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
