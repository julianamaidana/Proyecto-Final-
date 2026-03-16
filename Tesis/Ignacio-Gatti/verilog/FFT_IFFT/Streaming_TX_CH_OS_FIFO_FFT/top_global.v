`timescale 1ns/1ps
`default_nettype none

// ============================================================
// top_global_all  v9  (con xhist_delay)
//
// Cadena completa:
//   TX -> CH -> OS -> FIFO -> FFT -> HB -> CMUL -> IFFT -> DN -> SLICER -> ZPE -> FFT_ERROR
//                              └──── xhist_delay ────┘
//
// NUEVO en v9:
//   - Instancia xhist_delay conectado a hb_out_X_old_re/im
//   - Retrasa X_hist exactamente 118 ciclos para sincronizar con ffte_out
//   - Puertos nuevos: xhd_out_valid, xhd_out_start, xhd_out_re/im
//   - VERIFICACION: xhd_out_start debe coincidir con ffte_out_start
// ============================================================

module top_global_all #(
    parameter integer N_OS      = 16,
    parameter integer WN        = 9,
    parameter integer FIFO_AW   = 8,

    parameter integer NFFT      = 32,
    parameter integer LOGN      = 5,
    parameter integer NB_INT    = 17,
    parameter integer NBF_INT   = 10,
    parameter integer REORDER_BITREV = 1,

    parameter integer CH_GAIN_SH = 0,
    parameter integer K_HIST     = 1
)(
    input  wire                 clk_fast,
    input  wire                 rst,
    input  wire                 enable_div,
    input  wire [10:0]          sigma_scale,

    output wire                 clk_low,

    // --- Debug TX/CH ---
    output wire signed [WN-1:0] tx_I_dbg,
    output wire signed [WN-1:0] tx_Q_dbg,
    output wire signed [WN-1:0] ch_I_dbg,
    output wire signed [WN-1:0] ch_Q_dbg,

    // --- OS buffer ---
    output wire                 os_overflow,
    output wire                 os_start,
    output wire                 os_valid,
    output wire signed [WN-1:0] os_I,
    output wire signed [WN-1:0] os_Q,

    // --- FIFO ---
    output wire                 fifo_full,
    output wire                 fifo_empty,
    output wire                 fifo_overflow,
    output wire [FIFO_AW:0]     fifo_count,

    // --- Entrada a la FFT (debug) ---
    output wire                 fft_in_valid,
    output wire                 fft_in_start,
    output wire signed [WN-1:0] fft_in_I,
    output wire signed [WN-1:0] fft_in_Q,

    // --- Salida de la FFT ---
    output wire                 fft_out_valid,
    output wire                 fft_out_start,
    output wire signed [NB_INT-1:0] fft_out_I,
    output wire signed [NB_INT-1:0] fft_out_Q,

    // --- History Buffer ---
    output wire                     hb_out_valid,
    output wire                     hb_out_start,
    output wire signed [NB_INT-1:0] hb_out_curr_I,
    output wire signed [NB_INT-1:0] hb_out_curr_Q,
    output wire signed [NB_INT-1:0] hb_out_old_I,
    output wire signed [NB_INT-1:0] hb_out_old_Q,

    // --- CMUL ---
    output wire                     cmul_out_valid,
    output wire                     cmul_out_start,
    output wire signed [NB_INT-1:0] cmul_out_I,
    output wire signed [NB_INT-1:0] cmul_out_Q,

    // --- IFFT (cruda, para debug) ---
    output wire                 ifft_out_valid,
    output wire                 ifft_out_start,
    output wire signed [WN-1:0] ifft_out_I,
    output wire signed [WN-1:0] ifft_out_Q,

    // --- DISCARD_N: y_blk (muestras N..2N-1 del frame) ---
    output wire                 dn_out_valid,
    output wire                 dn_out_start,
    output wire signed [WN-1:0] dn_out_I,
    output wire signed [WN-1:0] dn_out_Q,

    // --- SLICER_QPSK ---
    output wire                 sl_out_valid,   // 1 durante N muestras por frame
    output wire                 sl_out_start,   // 1 en la primera muestra del frame
    output wire signed [WN-1:0] sl_out_yhat_I,  // símbolo decidido Re  (±QPSK_A)
    output wire signed [WN-1:0] sl_out_yhat_Q,  // símbolo decidido Im  (±QPSK_A)
    output wire signed [WN-1:0] sl_out_e_I,     // error Re  e = yhat - y
    output wire signed [WN-1:0] sl_out_e_Q,     // error Im  e = yhat - y

    // --- ZERO_PAD_ERROR ---
    output wire                 zpe_out_valid,  // 1 durante 2N muestras por frame
    output wire                 zpe_out_start,  // 1 en la primera muestra (primer cero)
    output wire signed [WN-1:0] zpe_out_eI,     // Re: 0 (primera mitad) | e (segunda mitad)
    output wire signed [WN-1:0] zpe_out_eQ,     // Im: 0 (primera mitad) | e (segunda mitad)

    // --- FFT_ERROR ---
    output wire                       ffte_out_valid, // 1 durante 2N muestras por frame
    output wire                       ffte_out_start, // 1 en la primera muestra del frame
    output wire signed [NB_INT-1:0]   ffte_out_I,     // espectro del error Re  E_k
    output wire signed [NB_INT-1:0]   ffte_out_Q,     // espectro del error Im  E_k

    // --- XHIST_DELAY ---
    // X_hist retrasado 118 ciclos — sincronizado con ffte_out
    // VERIFICAR: xhd_out_start debe coincidir con ffte_out_start
    output wire                        xhd_out_valid,  // 1 cuando hay dato válido
    output wire                        xhd_out_start,  // 1 en primera muestra del frame
    output wire signed [NB_INT-1:0]    xhd_out_re,     // X_hist Re retrasado
    output wire signed [NB_INT-1:0]    xhd_out_im      // X_hist Im retrasado
);

    // ============================================================
    // Clock divider
    // ============================================================
    clock_div2 u_div2 (
        .i_clk_fast(clk_fast),
        .i_enable  (enable_div),
        .o_clk_low (clk_low)
    );

    // ============================================================
    // TX  (clk_low)
    // ============================================================
    reg i_en_tx;
    always @(posedge clk_low) begin
        if (rst) i_en_tx <= 1'b0;
        else     i_en_tx <= 1'b1;
    end

    wire signed [WN-1:0] sI_tx, sQ_tx;

    top_tx u_tx (
        .clk   (clk_low),
        .reset (rst),
        .i_en  (i_en_tx),
        .sI_out(sI_tx),
        .sQ_out(sQ_tx)
    );

    assign tx_I_dbg = sI_tx;
    assign tx_Q_dbg = sQ_tx;

    // ============================================================
    // Canal  (clk_low)
    // ============================================================
    wire signed [WN-1:0] sI_ch, sQ_ch;

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

    // ============================================================
    // Ganancia opcional CH -> OS
    // ============================================================
    function signed [WN-1:0] sat_wn;
        input signed [WN+7:0] x;
        reg signed [WN-1:0] maxv, minv;
        begin
            maxv = {1'b0, {(WN-1){1'b1}}};
            minv = {1'b1, {(WN-1){1'b0}}};
            if      (x > $signed(maxv)) sat_wn = maxv;
            else if (x < $signed(minv)) sat_wn = minv;
            else                        sat_wn = x[WN-1:0];
        end
    endfunction

    wire signed [WN+7:0] sI_ch_ext = {{8{sI_ch[WN-1]}}, sI_ch};
    wire signed [WN+7:0] sQ_ch_ext = {{8{sQ_ch[WN-1]}}, sQ_ch};
    wire signed [WN+7:0] sI_ch_sh  = (CH_GAIN_SH > 0) ? (sI_ch_ext <<< CH_GAIN_SH) : sI_ch_ext;
    wire signed [WN+7:0] sQ_ch_sh  = (CH_GAIN_SH > 0) ? (sQ_ch_ext <<< CH_GAIN_SH) : sQ_ch_ext;
    wire signed [WN-1:0] sI_ch_os  = sat_wn(sI_ch_sh);
    wire signed [WN-1:0] sQ_ch_os  = sat_wn(sQ_ch_sh);

    // ============================================================
    // OS Buffer  (clk_low -> clk_fast)
    // ============================================================
    reg i_valid_low;
    always @(posedge clk_low) begin
        if (rst) i_valid_low <= 1'b0;
        else     i_valid_low <= 1'b1;
    end

    os_buffer #(
        .N (N_OS),
        .WN(WN)
    ) u_os (
        .i_clk_low (clk_low),
        .i_clk_fast(clk_fast),
        .i_rst     (rst),
        .i_valid   (i_valid_low),
        .i_i       (sI_ch_os),
        .i_q       (sQ_ch_os),
        .o_overflow(os_overflow),
        .o_start   (os_start),
        .o_valid   (os_valid),
        .o_i       (os_I),
        .o_q       (os_Q)
    );

    // ============================================================
    // FIFO  (clk_fast)
    // ============================================================
    wire [2*WN:0] fifo_din  = {os_start, os_I, os_Q};
    wire [2*WN:0] fifo_dout;
    wire fifo_rd_en = !fifo_empty;

    fifo #(
        .DATA_WIDTH(2*WN + 1),
        .ADDR_WIDTH(FIFO_AW)
    ) u_fifo (
        .clk       (clk_fast),
        .rst       (rst),
        .din       (fifo_din),
        .wr_en     (os_valid),
        .rd_en     (fifo_rd_en),
        .dout      (fifo_dout),
        .full      (fifo_full),
        .empty     (fifo_empty),
        .valid     (/* no usado */),
        .overflow  (fifo_overflow),
        .data_count(fifo_count)
    );

    wire rd_fire = fifo_rd_en && !fifo_empty;
    reg  rd_fire_q;
    always @(posedge clk_fast) begin
        if (rst) rd_fire_q <= 1'b0;
        else     rd_fire_q <= rd_fire;
    end

    assign fft_in_valid = rd_fire_q;
    assign fft_in_start = fifo_dout[2*WN];
    assign fft_in_I     = fifo_dout[2*WN-1:WN];
    assign fft_in_Q     = fifo_dout[WN-1:0];

    // ============================================================
    // FFT  (clk_fast)
    // ============================================================
    wire fft_rdy;

    fft_ifft_stream #(
        .NFFT(NFFT), .LOGN(LOGN),
        .NB_IN(WN),      .NBF_IN(7),
        .NB_W(NB_INT),   .NBF_W(NBF_INT),
        .NB_OUT(NB_INT), .NBF_OUT(NBF_INT),
        .BF_SCALE(0),
        .REORDER_BITREV(REORDER_BITREV)
    ) u_fft (
        .i_clk    (clk_fast),
        .i_rst    (rst),
        .i_valid  (fft_in_valid),
        .i_start  (fft_in_start),
        .i_xI     (fft_in_I),
        .i_xQ     (fft_in_Q),
        .i_inverse(1'b0),
        .o_in_ready(fft_rdy),
        .o_start  (fft_out_start),
        .o_valid  (fft_out_valid),
        .o_yI     (fft_out_I),
        .o_yQ     (fft_out_Q)
    );

    // ============================================================
    // HISTORY BUFFER  (clk_fast)
    // ============================================================
    wire [$clog2(K_HIST+1)-1:0] hb_wr_bank_dbg;
    wire [$clog2(NFFT)-1:0]     hb_samp_dbg;

    history_buffer #(
        .NB_W(NB_INT),
        .NFFT(NFFT),
        .K   (K_HIST)
    ) u_hb (
        .clk        (clk_fast),
        .rst        (rst),
        .i_valid    (fft_out_valid),
        .i_start    (fft_out_start),
        .i_xI       (fft_out_I),
        .i_xQ       (fft_out_Q),
        .o_valid    (hb_out_valid),
        .o_start    (hb_out_start),
        .o_X_curr_re(hb_out_curr_I),
        .o_X_curr_im(hb_out_curr_Q),
        .o_X_old_re (hb_out_old_I),
        .o_X_old_im (hb_out_old_Q),
        .o_wr_bank  (hb_wr_bank_dbg),
        .o_samp_idx (hb_samp_dbg)
    );

    // ============================================================
    // CMUL_PBFDAF  (clk_fast)
    //   Y[k] = W0[k]·X_curr[k] + W1[k]·X_old[k]
    //   Inicialización: W0=identidad, W1=0  →  Y = X_curr
    // ============================================================
    wire [$clog2(NFFT)-1:0] cmul_samp_dbg;

    cmul_pbfdaf #(
        .NB_W (NB_INT),
        .NBF_W(NBF_INT),
        .NFFT (NFFT)
    ) u_cmul (
        .clk       (clk_fast),
        .rst       (rst),
        .i_valid   (hb_out_valid),
        .i_start   (hb_out_start),
        .i_X0_re   (hb_out_curr_I),
        .i_X0_im   (hb_out_curr_Q),
        .i_X1_re   (hb_out_old_I),
        .i_X1_im   (hb_out_old_Q),
        // Puerto LMS inactivo hasta integrar UPDATE_LMS
        .i_we      (1'b0),
        .i_wk      ({$clog2(NFFT){1'b0}}),
        .i_wsel    (1'b0),
        .i_W_re    ({NB_INT{1'b0}}),
        .i_W_im    ({NB_INT{1'b0}}),
        .o_valid   (cmul_out_valid),
        .o_start   (cmul_out_start),
        .o_yI      (cmul_out_I),
        .o_yQ      (cmul_out_Q),
        .o_samp_idx(cmul_samp_dbg)
    );

    // ============================================================
    // IFFT  (clk_fast)
    // ============================================================
    wire ifft_rdy;

    fft_ifft_stream #(
        .NFFT(NFFT), .LOGN(LOGN),
        .NB_IN(NB_INT),  .NBF_IN(NBF_INT),
        .NB_W(NB_INT),   .NBF_W(NBF_INT),
        .NB_OUT(WN),     .NBF_OUT(7),
        .BF_SCALE(0),
        .REORDER_BITREV(REORDER_BITREV)
    ) u_ifft (
        .i_clk    (clk_fast),
        .i_rst    (rst),
        .i_valid  (cmul_out_valid),
        .i_start  (cmul_out_start),
        .i_xI     (cmul_out_I),
        .i_xQ     (cmul_out_Q),
        .i_inverse(1'b1),
        .o_in_ready(ifft_rdy),
        .o_start  (ifft_out_start),
        .o_valid  (ifft_out_valid),
        .o_yI     (ifft_out_I),
        .o_yQ     (ifft_out_Q)
    );

    // ============================================================
    // DISCARD_N  (clk_fast)
    //
    //   Recibe el stream IFFT de NFFT=32 muestras por frame.
    //   Descarta los primeros N=16 (transitorio overlap-save).
    //   Emite los últimos N=16 como y_blk: la salida útil del filtro.
    //
    //   Latencia añadida: 1 ciclo (registro de salida).
    //
    //   Próximo en cadena: SLICER_QPSK  (dn_out_* → slicer)
    // ============================================================
    discard_n #(
        .NB_W (WN),
        .NBF_W(7),
        .NFFT (NFFT)
    ) u_dn (
        .clk       (clk_fast),
        .rst       (rst),
        .i_valid   (ifft_out_valid),
        .i_start   (ifft_out_start),
        .i_yI      (ifft_out_I),
        .i_yQ      (ifft_out_Q),
        .o_valid   (dn_out_valid),
        .o_start   (dn_out_start),
        .o_yI      (dn_out_I),
        .o_yQ      (dn_out_Q),
        .o_samp_idx(/* debug no expuesto */)
    );

    // ============================================================
    // SLICER_QPSK  (u_slicer)
    //
    //   Entrada:  y_blk  = dn_out_*  (las N muestras útiles del frame)
    //   Salida:   yhat   = decisión dura QPSK  →  ±QPSK_A = ±91 en Q(9,7)
    //             e      = yhat - y            →  error para el LMS
    //
    //   Latencia: 1 ciclo (registro de salida).
    //   Próximo en cadena: FFT_ERROR (bloque LMS, no implementado aún).
    // ============================================================
    slicer_qpsk #(
        .NB_W (WN),
        .NBF_W(7),
        .NFFT (NFFT)
    ) u_slicer (
        .clk      (clk_fast),
        .rst      (rst),
        .i_valid  (dn_out_valid),
        .i_start  (dn_out_start),
        .i_yI     (dn_out_I),
        .i_yQ     (dn_out_Q),
        .o_valid  (sl_out_valid),
        .o_start  (sl_out_start),
        .o_yhat_I (sl_out_yhat_I),
        .o_yhat_Q (sl_out_yhat_Q),
        .o_e_I    (sl_out_e_I),
        .o_e_Q    (sl_out_e_Q)
    );

    // ============================================================
    // ZERO_PAD_ERROR  (u_zpe)
    //
    //   Entrada:  e_blk  = sl_out_e_*  (N errores del slicer)
    //   Salida:   frame  = [0...0 | e_blk]  (2N muestras para FFT_ERROR)
    //
    //   FSM interna: RECV(N) → ZEROS(N) → ERROR(N)
    //   Latencia: N ciclos de recepción antes de emitir el frame
    //   Próximo en cadena: FFT_ERROR (rama LMS).
    // ============================================================
    zero_pad_error #(
        .NB_W (WN),
        .NFFT (NFFT)
    ) u_zpe (
        .clk     (clk_fast),
        .rst     (rst),
        .i_valid (sl_out_valid),
        .i_start (sl_out_start),
        .i_eI    (sl_out_e_I),
        .i_eQ    (sl_out_e_Q),
        .o_valid (zpe_out_valid),
        .o_start (zpe_out_start),
        .o_eI    (zpe_out_eI),
        .o_eQ    (zpe_out_eQ)
    );

    // ============================================================
    // FFT_ERROR  (u_fft_error)
    //
    //   Entrada:  zpe_out_*  (frame 2N = [0..0 | e_blk])
    //   Salida:   E_k        (espectro del error, dominio frecuencia)
    //
    //   Reuso directo de fft_ifft_stream con i_inverse=0.
    //   Mismos parámetros que u_fft (cadena principal).
    //   Entrada en WN bits (9), salida en NB_INT bits (17).
    //   Próximo en cadena: GRADIENTE (PHI = conj(X_hist) * E_k).
    // ============================================================
    fft_ifft_stream #(
        .NFFT(NFFT), .LOGN(LOGN),
        .NB_IN(WN),      .NBF_IN(7),
        .NB_W(NB_INT),   .NBF_W(NBF_INT),
        .NB_OUT(NB_INT), .NBF_OUT(NBF_INT),
        .BF_SCALE(0),
        .REORDER_BITREV(REORDER_BITREV)
    ) u_fft_error (
        .i_clk    (clk_fast),
        .i_rst    (rst),
        .i_valid  (zpe_out_valid),
        .i_start  (zpe_out_start),
        .i_xI     (zpe_out_eI),
        .i_xQ     (zpe_out_eQ),
        .i_inverse(1'b0),
        .o_in_ready(),
        .o_start  (ffte_out_start),
        .o_valid  (ffte_out_valid),
        .o_yI     (ffte_out_I),
        .o_yQ     (ffte_out_Q)
    );

    // ============================================================
    // XHIST_DELAY  (u_xhd)
    //
    //   Retrasa X_hist (hb_out_X_old_re/im) exactamente 118 ciclos
    //   para sincronizarlo con la salida de la FFT_ERROR (ffte_out).
    //
    //   Por qué 118 ciclos:
    //     El history buffer emite X_hist junto con la FFT principal.
    //     El error E_k tarda 118 ciclos más en calcularse:
    //       CMUL + IFFT + DN + SLICER + ZPE + FFT_ERROR = 118 ciclos
    //     Sin este retardo, cuando E_k llega al GRADIENTE,
    //     X_hist del mismo frame ya pasó hace ~4 frames.
    //
    //   VERIFICACION en waveform:
    //     xhd_out_start y ffte_out_start deben aparecer en el
    //     MISMO ciclo exacto. Si hay diferencia, ajustar DELAY.
    //
    //   Próximo en cadena: GRADIENTE (PHI = conj(X_hist) * E_k)
    // ============================================================
    xhist_delay #(
        .NB_W (NB_INT),
        .DELAY(118)
    ) u_xhd (
        .clk    (clk_fast),
        .rst    (rst),
        .i_valid(hb_out_valid),
        .i_start(hb_out_start),
        .i_xre  (hb_out_old_I),
        .i_xim  (hb_out_old_Q),
        .o_valid(xhd_out_valid),
        .o_start(xhd_out_start),
        .o_xre  (xhd_out_re),
        .o_xim  (xhd_out_im)
    );

endmodule

`default_nettype wire
