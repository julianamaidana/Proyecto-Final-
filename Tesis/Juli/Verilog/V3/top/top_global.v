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

    output wire                        fft_valid_out,
    output wire signed [8:0]           fft_out_I,
    output wire signed [8:0]           fft_out_Q,

    output wire                        ifft_valid_out,
    output wire signed [8:0]           ifft_out_I,
    output wire signed [8:0]           ifft_out_Q,

    output wire                        fft_w_valid_out,
    output wire signed [NB_WIDE-1:0]   fft_w_out_I,
    output wire signed [NB_WIDE-1:0]   fft_w_out_Q,
    output wire                        ifft_w_valid_out,
    output wire signed [NB_WIDE-1:0]   ifft_w_out_I,
    output wire signed [NB_WIDE-1:0]   ifft_w_out_Q,

    output wire                        hb_valid_out,
    output wire [4:0]                  hb_k_idx,
    output wire signed [8:0]           hb_curr_I,
    output wire signed [8:0]           hb_curr_Q,
    output wire signed [8:0]           hb_old_I,
    output wire signed [8:0]           hb_old_Q
);

    // ============================================================
    // Generador de Fase (Toggle 1/2 clock para 50MHz)
    // ============================================================
    reg phase;
    always @(posedge clk or posedge rst) begin
        if (rst) phase <= 1'b0;
        else     phase <= ~phase;
    end

    // Señales de la FIFO y Control
    wire fifo_full, fifo_empty;
    wire [17:0] fifo_dout;
    wire fifo_rd_en;
    wire fifo_data_valid; 
    wire os_in_ready_w;
    wire fft_in_ready_w;
    wire ifft_in_ready_w;

    // ============================================================
    // TRANSMISOR (PRBS)
    // ============================================================
    // Solo avanza si hay espacio en la FIFO y es su fase (50MHz)
    wire tx_en_w = phase & !fifo_full & !bypass_tx;
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

    // Fuente de datos (Test data o PRBS)
    wire signed [DWIDTH-1:0] to_fifo_I = (bypass_tx) ? test_data_I : tx_I;
    wire signed [DWIDTH-1:0] to_fifo_Q = (bypass_tx) ? test_data_Q : tx_Q;

    // ============================================================
    // FIFO DE XILINX (IP CORE)
    // ============================================================
    fifo_generator_0 u_fifo (
        .clk   (clk),
        .srst   (rst),
        .din   ({to_fifo_I, to_fifo_Q}), 
        .wr_en (phase & !fifo_full),      // Escribimos siempre a 50MHz
        .rd_en (fifo_rd_en),              // Pedido por el os_buffer
        .dout  (fifo_dout),
        .full  (fifo_full),
        .empty (fifo_empty),
        .valid (fifo_data_valid)          
    );

    // ============================================================
    // CANAL (Se procesa al salir de la FIFO)
    // ============================================================
    wire signed [DWIDTH-1:0] rx_fifo_I = fifo_dout[17:9];
    wire signed [DWIDTH-1:0] rx_fifo_Q = fifo_dout[8:0];

    wire signed [DWIDTH-1:0] ch_real_I, ch_real_Q;

    top_ch #(
        .DWIDTH     (DWIDTH),
        .SNR_WIDTH  (SNR_WIDTH)
    ) u_ch (
        .clk         (clk),
        .rst         (rst),
        .In_I        (rx_fifo_I),
        .In_Q        (rx_fifo_Q),
        .sigma_scale (sigma_scale),
        .Out_I       (ch_real_I),
        .Out_Q       (ch_real_Q)
    );

    assign ch_out_I = (IDEAL_CH) ? rx_fifo_I : ch_real_I;
    assign ch_out_Q = (IDEAL_CH) ? rx_fifo_Q : ch_real_Q;

    // ============================================================
    // RECEPTOR (OS_BUFFER)
    // ============================================================
    wire signed [DWIDTH-1:0] os_out_I, os_out_Q;
    wire                     os_valid_w;
    wire                     os_start_w;

    os_buffer #(
        .N  (N_PART),
        .WN (DWIDTH)
    ) u_os (
        .i_clk       (clk),
        .i_rst       (rst),
        .i_valid     (fifo_data_valid), // Dato válido desde la FIFO
        .i_xI        (ch_out_I),
        .i_xQ        (ch_out_Q),
        .i_fft_ready (fft_in_ready_w),  // Control de flujo hacia adelante
        .o_in_ready  (fifo_rd_en),      // Habilita lectura de la FIFO
        .o_fft_start (os_start_w),
        .o_fft_valid (os_valid_w),
        .o_fft_xI    (os_out_I),
        .o_fft_xQ    (os_out_Q)
    );

    // ============================================================
    // FFT WIDE (22 bits)
    // ============================================================
    wire fft_start_w;
    wire fft_w_valid;
    wire signed [NB_WIDE-1:0] fft_w_I, fft_w_Q;

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

    // sat_trunc para salida 9b (History)
    sat_trunc #(.NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE), .NB_XO(9), .NBF_XO(7), .ROUND_EVEN(1)) 
    u_fft9_I (.i_data(fft_w_I), .o_data(fft_out_I));

    sat_trunc #(.NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE), .NB_XO(9), .NBF_XO(7), .ROUND_EVEN(1)) 
    u_fft9_Q (.i_data(fft_w_Q), .o_data(fft_out_Q));

    assign fft_valid_out = fft_w_valid;

    // ============================================================
    // IFFT WIDE
    // ============================================================
    wire ifft_start_w;
    wire ifft_w_valid;
    wire signed [NB_WIDE-1:0] ifft_w_I, ifft_w_Q;

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

    // sat_trunc para salida debug 9b
    sat_trunc #(.NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE), .NB_XO(9), .NBF_XO(7), .ROUND_EVEN(1)) 
    u_ifft9_I (.i_data(ifft_w_I), .o_data(ifft_out_I));

    sat_trunc #(.NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE), .NB_XO(9), .NBF_XO(7), .ROUND_EVEN(1)) 
    u_ifft9_Q (.i_data(ifft_w_Q), .o_data(ifft_out_Q));

    assign ifft_valid_out = ifft_w_valid;

    // ============================================================
    // HISTORY BUFFER
    // ============================================================
    history_buffer #(.W(22)) u_history (
        .clk          (clk),
        .rst          (rst),
        .i_valid      (fft_w_valid),
        .i_X_re       (fft_w_I),
        .i_X_im       (fft_w_Q),
        .i_W0_re      (9'sd0), .i_W0_im(9'sd0),
        .i_W1_re      (9'sd0), .i_W1_im(9'sd0),
        .o_valid_data (hb_valid_out),
        .o_k_idx      (hb_k_idx),
        .o_X_curr_re  (hb_curr_I),
        .o_X_curr_im  (hb_curr_Q),
        .o_X_old_re   (hb_old_I),
        .o_X_old_im   (hb_old_Q)
    );

endmodule