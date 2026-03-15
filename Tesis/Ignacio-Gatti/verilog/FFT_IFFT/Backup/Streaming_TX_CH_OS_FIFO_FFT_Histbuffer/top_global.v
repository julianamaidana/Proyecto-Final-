`timescale 1ns/1ps
`default_nettype none

// ============================================================
// top_global_all  v2  (con history_buffer)
//
// Cadena completa:
//   TX(clk_low) -> CH(clk_low)
//   -> OS_BUFFER(clk_low -> clk_fast)
//   -> FIFO(clk_fast)
//   -> FFT(clk_fast)           [NB_INT=17 bits salida]
//   -> HISTORY_BUFFER(clk_fast)[almacena K frames pasados de la FFT]
//   -> IFFT(clk_fast)          [recibe X_curr del history buffer]
//
// Dominio de relojes:
//   clk_low  = clk_fast / 2  (TX, CH, valid de OS)
//   clk_fast                 (OS salida, FIFO, FFT, HB, IFFT)
//
// Parámetro K_HIST:
//   Cantidad de bloques pasados conservados en el history_buffer.
//   Para PBFDAF con 1 bloque anterior usá K_HIST=1 (default).
//   Aumentar K_HIST agrega latencia (K ciclos de frame = K*NFFT clocks)
//   pero NO cambia el throughput: el streaming se mantiene continuo.
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

    // ============================================================
    // History Buffer: bloques pasados a conservar
    // ============================================================
    parameter integer K_HIST    = 1
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

    // --- History Buffer: frame actual (pasado a IFFT) ---
    output wire                     hb_out_valid,
    output wire                     hb_out_start,
    output wire signed [NB_INT-1:0] hb_out_curr_I,
    output wire signed [NB_INT-1:0] hb_out_curr_Q,

    // --- History Buffer: frame pasado (para ecualizador futuro) ---
    output wire signed [NB_INT-1:0] hb_out_old_I,
    output wire signed [NB_INT-1:0] hb_out_old_Q,

    // --- Salida IFFT ---
    output wire                 ifft_out_valid,
    output wire                 ifft_out_start,
    output wire signed [WN-1:0] ifft_out_I,
    output wire signed [WN-1:0] ifft_out_Q
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
        reg signed [WN-1:0] maxv;
        reg signed [WN-1:0] minv;
        begin
            maxv = {1'b0, {(WN-1){1'b1}}};
            minv = {1'b1, {(WN-1){1'b0}}};
            if (x > $signed(maxv))      sat_wn = maxv;
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
    //   Empaqueta {start, I, Q} en un único word
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

    // Valid alineado con fifo_dout (latencia 1 ciclo de FIFO)
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
    //   Salida: NB_INT bits (ancho interno, punto fijo NBF_INT)
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
    //
    //   Entrada: salida directa de la FFT
    //   Salida:
    //     o_X_curr -> frame actual  -> alimenta IFFT (bypass limpio)
    //     o_X_old  -> frame n-K_HIST -> disponible para ecualizador
    //
    //   Latencia introducida: 1 ciclo de registro (lectura RAM sync)
    //   El o_valid/o_start salen 1 ciclo después de fft_out_valid/start.
    //
    //   IMPORTANTE para PBFDAF:
    //   En esta etapa el history buffer es pasado a través (pass-through)
    //   hacia la IFFT usando X_curr. X_old queda expuesto como puerto
    //   de salida del top para que la próxima etapa (multiplicador de
    //   pesos, filtro frecuencial) lo use sin modificar este módulo.
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
    // IFFT  (clk_fast)
    //   Recibe X_curr del history buffer.
    //   Cuando el ecualizador esté integrado, la señal que entra
    //   aquí será la salida del filtro frecuencial (W*X), no X_curr.
    //   Por ahora: IFFT(FFT(x)) = x  (identidad).
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
        .i_valid  (hb_out_valid),    // <-- viene del HB, no directo de FFT
        .i_start  (hb_out_start),    // <-- viene del HB
        .i_xI     (hb_out_curr_I),   // <-- X_curr del HB
        .i_xQ     (hb_out_curr_Q),
        .i_inverse(1'b1),
        .o_in_ready(ifft_rdy),
        .o_start  (ifft_out_start),
        .o_valid  (ifft_out_valid),
        .o_yI     (ifft_out_I),
        .o_yQ     (ifft_out_Q)
    );

endmodule

`default_nettype wire
