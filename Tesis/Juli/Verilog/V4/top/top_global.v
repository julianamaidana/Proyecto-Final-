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
    // 1. GENERADOR DE FASE (Clock Enable 50MHz)
    // ============================================================
    reg phase;
    always @(posedge clk) begin
        if (rst) phase <= 1'b0;
        else     phase <= ~phase;
    end

    // Señales de interconexión
    wire signed [DWIDTH-1:0] tx_I, tx_Q;
    wire signed [DWIDTH-1:0] ch_real_I, ch_real_Q;
    wire [DWIDTH-1:0]        fifo_out_I, fifo_out_Q;
    wire                     fifo_full, fifo_empty;
    wire                     fifo_rd_en;
    wire                     fifo_data_valid; // <--- SEÑAL CRÍTICA DE ALINEACIÓN
    wire                     os_start_w, os_valid_w;
    wire [DWIDTH-1:0]        os_out_I, os_out_Q;
    wire                     fft_in_ready_w;

    // ============================================================
    // 2. TRANSMISOR (PRBS) - Sistema Real (No se frena)
    // ============================================================
    top_tx u_tx (
        .clk   (clk),
        .reset (rst),
        .i_en  (phase & !bypass_tx), // Solo depende de la fase, no del estado de la FIFO
        .sI_out(tx_I),
        .sQ_out(tx_Q)
    );

    assign tx_sym_I = tx_I;
    assign tx_sym_Q = tx_Q;

    // ============================================================
    // 3. CANAL (Procesamiento)
    // ============================================================
    top_ch #(
        .DWIDTH     (DWIDTH),
        .SNR_WIDTH  (SNR_WIDTH)
    ) u_ch (
        .clk         (clk),
        .rst         (rst),
        .In_I        (bypass_tx ? test_data_I : tx_I),
        .In_Q        (bypass_tx ? test_data_Q : tx_Q),
        .sigma_scale (sigma_scale),
        .Out_I       (ch_real_I),
        .Out_Q       (ch_real_Q) 
    );

    assign ch_out_I = IDEAL_CH ? (bypass_tx ? test_data_I : tx_I) : ch_real_I;
    assign ch_out_Q = IDEAL_CH ? (bypass_tx ? test_data_Q : tx_Q) : ch_real_Q;

    // ============================================================
    // 4. TU FIFO PROPIA (Buffer de desacople)
    // ============================================================
    fifo #(
        .DATA_WIDTH (DWIDTH * 2), 
        .ADDR_WIDTH (8) // 256 posiciones
    ) u_fifo_inst (
        .clk      (clk),
        .rst      (rst),
        .din      ({ch_out_I, ch_out_Q}),
        .wr_en    (phase),              // Escribimos cada vez que el canal procesa
        .rd_en    (fifo_rd_en),         // Pedido por el os_buffer (o_in_ready)
        .dout     ({fifo_out_I, fifo_out_Q}),
        .full     (fifo_full),
        .empty    (fifo_empty),
        .valid    (fifo_data_valid),    // <--- Conectada para asegurar alineación
        .overflow ()                    // Aquí verás si el sistema pierde datos
    );

    // ============================================================
    // 5. RECEPTOR (OS_BUFFER) - Ahora con Handshake correcto
    // ============================================================
    os_buffer #(
        .N  (N_PART),
        .WN (DWIDTH)
    ) u_os (
        .i_clk       (clk),
        .i_rst       (rst),
        .i_valid     (fifo_data_valid), // <--- Solo captura cuando el dato está en el bus
        .i_xI        (fifo_out_I),
        .i_xQ        (fifo_out_Q),
        .o_in_ready  (fifo_rd_en),      // Controla el rd_en de la FIFO
        .o_fft_start (os_start_w),
        .o_fft_valid (os_valid_w),
        .o_fft_xI    (os_out_I),
        .o_fft_xQ    (os_out_Q) 
    );

    // ============================================================
    // 6. FFT WIDE
    // ============================================================
    wire signed [NB_WIDE-1:0] fft_w_I, fft_w_Q;
    wire fft_w_valid;

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
        .o_start    (), 
        .o_valid    (fft_w_valid),
        .o_yI       (fft_w_I),
        .o_yQ       (fft_w_Q) 
    );

    assign fft_w_valid_out = fft_w_valid;
    assign fft_w_out_I     = fft_w_I;
    assign fft_w_out_Q     = fft_w_Q;

    // Saturation and Truncation
    sat_trunc #(.NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE), .NB_XO(9), .NBF_XO(7), .ROUND_EVEN(1)) 
        u_fft9_I (.i_data(fft_w_I), .o_data(fft_out_I));
    sat_trunc #(.NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE), .NB_XO(9), .NBF_XO(7), .ROUND_EVEN(1)) 
        u_fft9_Q (.i_data(fft_w_Q), .o_data(fft_out_Q));

    assign fft_valid_out = fft_w_valid;

    // ============================================================
    // 6. IFFT WIDE [cite: 67, 68]
    // ============================================================
    wire signed [NB_WIDE-1:0] ifft_w_I, ifft_w_Q;
    wire ifft_w_valid;

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
        .i_clk      (clk), [cite: 69]
        .i_rst      (rst), [cite: 69]
        .i_valid    (fft_w_valid), [cite: 69]
        .i_xI       (fft_w_I), [cite: 69]
        .i_xQ       (fft_w_Q), [cite: 70]
        .i_inverse  (1'b1), [cite: 70]
        .o_in_ready (),
        .o_start    (),
        .o_valid    (ifft_w_valid), [cite: 70]
        .o_yI       (ifft_w_I), [cite: 70]
        .o_yQ       (ifft_w_Q)  [cite: 70]
    );

    assign ifft_w_valid_out = ifft_w_valid; [cite: 71]
    assign ifft_w_out_I     = ifft_w_I; [cite: 71]
    assign ifft_w_out_Q     = ifft_w_Q; [cite: 71]

    sat_trunc #(.NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE), .NB_XO(9), .NBF_XO(7), .ROUND_EVEN(1)) 
        u_ifft9_I (.i_data(ifft_w_I), .o_data(ifft_out_I)); [cite: 72]
    sat_trunc #(.NB_XI(NB_WIDE), .NBF_XI(NBF_WIDE), .NB_XO(9), .NBF_XO(7), .ROUND_EVEN(1)) 
        u_ifft9_Q (.i_data(ifft_w_Q), .o_data(ifft_out_Q)); [cite: 73]

    assign ifft_valid_out = ifft_w_valid; [cite: 73]

    // ============================================================
    // 7. HISTORY BUFFER [cite: 74]
    // ============================================================
    history_buffer #(.W(22)) u_history (
        .clk          (clk), [cite: 74]
        .rst          (rst), [cite: 74]
        .i_valid      (fft_w_valid), [cite: 74]
        .i_X_re       (fft_w_I), [cite: 74]
        .i_X_im       (fft_w_Q), [cite: 74]
        .i_W0_re      (9'sd0), .i_W0_im(9'sd0), [cite: 75]
        .i_W1_re      (9'sd0), .i_W1_im(9'sd0), [cite: 75]
        .o_valid_data (hb_valid_out), [cite: 75]
        .o_k_idx      (hb_k_idx), [cite: 75]
        .o_X_curr_re  (hb_curr_I), [cite: 75]
        .o_X_curr_im  (hb_curr_Q), [cite: 75]
        .o_X_old_re   (hb_old_I), [cite: 75]
        .o_X_old_im   (hb_old_Q)  [cite: 75]
    );

endmodule