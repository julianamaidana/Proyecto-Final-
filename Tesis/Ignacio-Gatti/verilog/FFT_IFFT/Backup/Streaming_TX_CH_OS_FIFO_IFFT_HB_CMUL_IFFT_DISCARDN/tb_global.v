`timescale 1ns/1ps
`default_nettype none

// ============================================================
// tb_top_global_all  v5  (con discard_n)
//
// Verifica la cadena completa:
//   TX -> CH -> OS -> FIFO -> FFT -> HB -> CMUL -> IFFT -> DISCARD_N
//
// BLOQUES DE VERIFICACIÓN
// -----------------------
// [BLOQUE 1] OS   : overflow y periodo entre starts
// [BLOQUE 2] FIFO : overflow y ocupación máxima
// [BLOQUE 3] FFT  : frames procesados y periodo de start
// [BLOQUE 4] HB   : X_curr == fft_out[t-1],  X_old == fft_out[t-1-K*NFFT]
// [BLOQUE 5] CMUL : con W=identidad, cmul_out == hb_out_curr[t-1]
// [BLOQUE 6] IFFT : IFFT(FFT(x)) == x  (tolerancia TOL)
//
// [BLOQUE 7] DISCARD_N  <-- NUEVO en v5
//   Estrategia — delay line directa (sin for en always):
//     discard_n tiene 1 ciclo de latencia. dn_out[T] == ifft_out[T-1].
//     Se registra ifft_out 1 ciclo (ifft_d1) y se compara directamente.
//   P1) N=16 salidas válidas por frame.  P2) 1 start por frame.
//   P4) dn_out[ciclo T] == ifft_out[ciclo T-1].
// ============================================================

module tb_top_global_all;

    // ============================================================
    // Parámetros
    // ============================================================
    localparam integer WN        = 9;
    localparam integer NFFT      = 32;
    localparam integer N_HALF    = NFFT / 2;   // 16
    localparam integer NB_INT    = 17;
    localparam integer FIFO_AW   = 8;
    localparam integer K_HIST    = 1;

    localparam integer TOL              = 3;
    localparam integer FRAMES_TO_CHECK  = 30;

    localparam integer QDEPTH = 128;
    localparam integer QMEM   = QDEPTH * NFFT;

    // ============================================================
    // Relojes y reset
    // ============================================================
    reg clk_fast;
    initial begin clk_fast = 1'b0; forever #5 clk_fast = ~clk_fast; end

    reg rst;
    initial begin
        rst = 1'b1;
        repeat(20) @(posedge clk_fast);
        rst = 1'b0;
    end

    reg enable_div;
    initial enable_div = 1'b1;

    reg [10:0] sigma_scale;
    initial sigma_scale = 11'd0;

    // ============================================================
    // Señales DUT
    // ============================================================
    wire clk_low;
    wire signed [WN-1:0]     tx_I_dbg, tx_Q_dbg;
    wire signed [WN-1:0]     ch_I_dbg, ch_Q_dbg;

    wire                     os_overflow, os_start, os_valid;
    wire signed [WN-1:0]     os_I, os_Q;

    wire                     fifo_full, fifo_empty, fifo_overflow;
    wire [FIFO_AW:0]         fifo_count;

    wire                     fft_in_valid, fft_in_start;
    wire signed [WN-1:0]     fft_in_I, fft_in_Q;

    wire                     fft_out_valid, fft_out_start;
    wire signed [NB_INT-1:0] fft_out_I, fft_out_Q;

    wire                     hb_out_valid, hb_out_start;
    wire signed [NB_INT-1:0] hb_out_curr_I, hb_out_curr_Q;
    wire signed [NB_INT-1:0] hb_out_old_I,  hb_out_old_Q;

    wire                     cmul_out_valid, cmul_out_start;
    wire signed [NB_INT-1:0] cmul_out_I, cmul_out_Q;

    wire                     ifft_out_valid, ifft_out_start;
    wire signed [WN-1:0]     ifft_out_I, ifft_out_Q;

    // DISCARD_N (nuevo en v5)
    wire                     dn_out_valid, dn_out_start;
    wire signed [WN-1:0]     dn_out_I, dn_out_Q;

    // ============================================================
    // DUT
    // ============================================================
    top_global_all #(
        .N_OS   (16),
        .WN     (WN),
        .FIFO_AW(FIFO_AW),
        .NFFT   (NFFT),
        .NB_INT (NB_INT),
        .K_HIST (K_HIST)
    ) dut (
        .clk_fast      (clk_fast),
        .rst           (rst),
        .enable_div    (enable_div),
        .sigma_scale   (sigma_scale),
        .clk_low       (clk_low),
        .tx_I_dbg      (tx_I_dbg),
        .tx_Q_dbg      (tx_Q_dbg),
        .ch_I_dbg      (ch_I_dbg),
        .ch_Q_dbg      (ch_Q_dbg),
        .os_overflow   (os_overflow),
        .os_start      (os_start),
        .os_valid      (os_valid),
        .os_I          (os_I),
        .os_Q          (os_Q),
        .fifo_full     (fifo_full),
        .fifo_empty    (fifo_empty),
        .fifo_overflow (fifo_overflow),
        .fifo_count    (fifo_count),
        .fft_in_valid  (fft_in_valid),
        .fft_in_start  (fft_in_start),
        .fft_in_I      (fft_in_I),
        .fft_in_Q      (fft_in_Q),
        .fft_out_valid (fft_out_valid),
        .fft_out_start (fft_out_start),
        .fft_out_I     (fft_out_I),
        .fft_out_Q     (fft_out_Q),
        .hb_out_valid  (hb_out_valid),
        .hb_out_start  (hb_out_start),
        .hb_out_curr_I (hb_out_curr_I),
        .hb_out_curr_Q (hb_out_curr_Q),
        .hb_out_old_I  (hb_out_old_I),
        .hb_out_old_Q  (hb_out_old_Q),
        .cmul_out_valid(cmul_out_valid),
        .cmul_out_start(cmul_out_start),
        .cmul_out_I    (cmul_out_I),
        .cmul_out_Q    (cmul_out_Q),
        .ifft_out_valid(ifft_out_valid),
        .ifft_out_start(ifft_out_start),
        .ifft_out_I    (ifft_out_I),
        .ifft_out_Q    (ifft_out_Q),
        .dn_out_valid  (dn_out_valid),
        .dn_out_start  (dn_out_start),
        .dn_out_I      (dn_out_I),
        .dn_out_Q      (dn_out_Q)
    );

    // ============================================================
    // Funciones auxiliares
    // ============================================================
    function integer iabs;
        input integer v;
        begin iabs = (v < 0) ? -v : v; end
    endfunction

    function integer wrap_inc;
        input integer ptr;
        begin wrap_inc = (ptr == (QDEPTH-1)) ? 0 : (ptr + 1); end
    endfunction

    // ============================================================
    // Contador de ciclos global
    // ============================================================
    integer cyc;
    initial cyc = 0;
    always @(posedge clk_fast) begin
        if (rst) cyc <= 0;
        else     cyc <= cyc + 1;
    end

    // ============================================================
    // BLOQUE 1 — OS Monitor
    // ============================================================
    integer last_os_start_cyc, os_warns;
    initial begin last_os_start_cyc = -1; os_warns = 0; end

    always @(posedge clk_fast) begin
        if (rst) begin last_os_start_cyc <= -1; os_warns <= 0; end
        else if (os_valid && os_start) begin
            if (last_os_start_cyc >= 0 && (cyc - last_os_start_cyc) != NFFT) begin
                os_warns <= os_warns + 1;
                $display("[%0t] WARN OS: periodo=%0d (exp %0d)",
                    $time, cyc-last_os_start_cyc, NFFT);
            end
            last_os_start_cyc <= cyc;
        end
    end

    // ============================================================
    // BLOQUE 2 — FIFO Monitor
    // ============================================================
    integer fifo_max_count;
    initial fifo_max_count = 0;
    always @(posedge clk_fast) begin
        if (rst) fifo_max_count <= 0;
        else if ($signed(fifo_count) > fifo_max_count)
            fifo_max_count <= fifo_count;
    end

    // ============================================================
    // BLOQUE 3 — FFT Monitor
    // ============================================================
    integer fft_frames_out, last_fft_start_cyc, fft_period_warns;
    initial begin fft_frames_out=0; last_fft_start_cyc=-1; fft_period_warns=0; end

    always @(posedge clk_fast) begin
        if (rst) begin fft_frames_out<=0; last_fft_start_cyc<=-1; fft_period_warns<=0; end
        else if (fft_out_valid && fft_out_start) begin
            fft_frames_out <= fft_frames_out + 1;
            if (last_fft_start_cyc >= 0 && (cyc - last_fft_start_cyc) != NFFT) begin
                fft_period_warns <= fft_period_warns + 1;
                $display("[%0t] WARN FFT: periodo=%0d", $time, cyc-last_fft_start_cyc);
            end
            last_fft_start_cyc <= cyc;
        end
    end

    // ============================================================
    // BLOQUE 4 — History Buffer Monitor
    // ============================================================
    localparam integer DLY_DEPTH = K_HIST * NFFT + 1;

    reg signed [NB_INT-1:0] fft_dly_I [0:DLY_DEPTH-1];
    reg signed [NB_INT-1:0] fft_dly_Q [0:DLY_DEPTH-1];
    integer fft_dly_fill;

    reg signed [NB_INT-1:0] fft_q_I [0:QMEM-1];
    reg signed [NB_INT-1:0] fft_q_Q [0:QMEM-1];
    integer fft_wr_ptr, fft_q_count, fft_in_samp;

    integer hb_total_curr_errs, hb_total_old_errs;
    integer hb_frame_errs_curr, hb_frame_errs_old;
    integer hb_frame_total;

    reg signed [NB_INT-1:0] fft_prev_I,  fft_prev_Q;
    reg                      fft_prev_valid;

    integer dly_ii;
    initial begin
        for (dly_ii=0; dly_ii<DLY_DEPTH; dly_ii=dly_ii+1) begin
            fft_dly_I[dly_ii]=0; fft_dly_Q[dly_ii]=0;
        end
        fft_dly_fill=0; fft_wr_ptr=0; fft_q_count=0; fft_in_samp=0;
        hb_total_curr_errs=0; hb_total_old_errs=0;
        hb_frame_errs_curr=0; hb_frame_errs_old=0; hb_frame_total=0;
        fft_prev_valid=0; fft_prev_I=0; fft_prev_Q=0;
    end

    integer sh_i;
    always @(posedge clk_fast) begin : hb_dly
        integer qi_wr;
        if (rst) begin
            fft_dly_fill<=0; fft_wr_ptr<=0; fft_q_count<=0; fft_in_samp<=0;
            fft_prev_valid<=0;
        end else begin
            fft_prev_valid <= fft_out_valid;
            fft_prev_I     <= fft_out_I;
            fft_prev_Q     <= fft_out_Q;

            if (fft_out_valid) begin
                for (sh_i=DLY_DEPTH-1; sh_i>0; sh_i=sh_i-1) begin
                    fft_dly_I[sh_i] <= fft_dly_I[sh_i-1];
                    fft_dly_Q[sh_i] <= fft_dly_Q[sh_i-1];
                end
                fft_dly_I[0] <= fft_out_I;
                fft_dly_Q[0] <= fft_out_Q;
                if (fft_dly_fill < DLY_DEPTH) fft_dly_fill <= fft_dly_fill + 1;

                if (fft_out_start) fft_in_samp <= 0;
                qi_wr = (fft_wr_ptr % QDEPTH)*NFFT + fft_in_samp;
                fft_q_I[qi_wr] <= fft_out_I;
                fft_q_Q[qi_wr] <= fft_out_Q;
                if (fft_in_samp == NFFT-1) begin
                    fft_wr_ptr  <= wrap_inc(fft_wr_ptr);
                    fft_q_count <= (fft_q_count < QDEPTH) ? fft_q_count+1 : fft_q_count;
                    fft_in_samp <= 0;
                end else fft_in_samp <= fft_in_samp + 1;
            end
        end
    end

    always @(posedge clk_fast) begin : hb_chk
        integer dIc, dQc, dIo, dQo;
        if (rst) begin
            hb_total_curr_errs<=0; hb_total_old_errs<=0;
            hb_frame_errs_curr<=0; hb_frame_errs_old<=0; hb_frame_total<=0;
        end else if (hb_out_valid) begin
            if (hb_out_start) begin
                if (hb_frame_total>0) begin
                    if (hb_frame_errs_curr==0 && hb_frame_errs_old==0)
                        $display("[HB][FRAME %0d] PASS  curr_errs=0  old_errs=0", hb_frame_total-1);
                    else
                        $display("[HB][FRAME %0d] FAIL  curr_errs=%0d  old_errs=%0d",
                            hb_frame_total-1, hb_frame_errs_curr, hb_frame_errs_old);
                end
                hb_frame_errs_curr<=0; hb_frame_errs_old<=0;
                hb_frame_total<=hb_frame_total+1;
            end else begin
                dIc = $signed(hb_out_curr_I) - $signed(fft_dly_I[0]);
                dQc = $signed(hb_out_curr_Q) - $signed(fft_dly_Q[0]);
                if (iabs(dIc)>TOL || iabs(dQc)>TOL) begin
                    hb_frame_errs_curr<=hb_frame_errs_curr+1;
                    hb_total_curr_errs<=hb_total_curr_errs+1;
                end
                if (fft_dly_fill >= DLY_DEPTH) begin
                    dIo = $signed(hb_out_old_I) - $signed(fft_dly_I[K_HIST*NFFT]);
                    dQo = $signed(hb_out_old_Q) - $signed(fft_dly_Q[K_HIST*NFFT]);
                    if (iabs(dIo)>TOL || iabs(dQo)>TOL) begin
                        hb_frame_errs_old<=hb_frame_errs_old+1;
                        hb_total_old_errs<=hb_total_old_errs+1;
                    end
                end
            end
        end
    end

    // ============================================================
    // BLOQUE 5 — CMUL Monitor
    // ============================================================
    reg signed [NB_INT-1:0] hb_prev_I, hb_prev_Q;
    reg                      hb_prev_valid, hb_prev_start;
    integer cmul_total_errs, cmul_frames_checked;
    integer cmul_frame_errs, cmul_start_misalign;

    initial begin
        hb_prev_I=0; hb_prev_Q=0; hb_prev_valid=0; hb_prev_start=0;
        cmul_total_errs=0; cmul_frames_checked=0;
        cmul_frame_errs=0; cmul_start_misalign=0;
    end

    always @(posedge clk_fast) begin : cmul_monitor
        integer dIcm, dQcm;
        if (rst) begin
            hb_prev_I<=0; hb_prev_Q<=0; hb_prev_valid<=0; hb_prev_start<=0;
            cmul_total_errs<=0; cmul_frames_checked<=0;
            cmul_frame_errs<=0; cmul_start_misalign<=0;
        end else begin
            hb_prev_valid <= hb_out_valid;
            hb_prev_start <= hb_out_start;
            hb_prev_I     <= hb_out_curr_I;
            hb_prev_Q     <= hb_out_curr_Q;

            if (cmul_out_valid) begin
                if (cmul_out_start && !hb_prev_start) begin
                    cmul_start_misalign <= cmul_start_misalign + 1;
                    $display("[CMUL][WARN] start desalineado frame %0d", cmul_frames_checked);
                end
                if (cmul_out_start) begin
                    if (cmul_frames_checked>0) begin
                        if (cmul_frame_errs==0)
                            $display("[CMUL][FRAME %0d] PASS  errs=0", cmul_frames_checked-1);
                        else
                            $display("[CMUL][FRAME %0d] FAIL  errs=%0d",
                                cmul_frames_checked-1, cmul_frame_errs);
                    end
                    cmul_frame_errs<=0;
                    cmul_frames_checked<=cmul_frames_checked+1;
                end
                if (hb_prev_valid) begin
                    dIcm = $signed(cmul_out_I) - $signed(hb_prev_I);
                    dQcm = $signed(cmul_out_Q) - $signed(hb_prev_Q);
                    if (iabs(dIcm)>TOL || iabs(dQcm)>TOL) begin
                        cmul_frame_errs<=cmul_frame_errs+1;
                        cmul_total_errs<=cmul_total_errs+1;
                        $display("[CMUL][ERR] frame=%0d dI=%0d dQ=%0d",
                            cmul_frames_checked-1, dIcm, dQcm);
                    end
                end
            end
        end
    end

    // ============================================================
    // BLOQUE 6 — IFFT identidad check
    // ============================================================
    reg signed [WN-1:0] in_mem_I [0:QMEM-1];
    reg signed [WN-1:0] in_mem_Q [0:QMEM-1];

    integer wr_ptr, rd_ptr, qcount;
    integer in_samp;
    integer checking, out_samp;
    integer checked_frames, frame_errs, total_errs;
    integer idx_in, idx_out, dI, dQ;

    initial begin
        wr_ptr=0; rd_ptr=0; qcount=0; in_samp=0;
        checking=0; out_samp=0;
        checked_frames=0; frame_errs=0; total_errs=0;
    end

    always @(posedge clk_fast) begin
        if (rst) begin wr_ptr<=0; qcount<=0; in_samp<=0; end
        else if (fft_in_valid) begin
            idx_in = fft_in_start ? 0 : in_samp;
            in_mem_I[wr_ptr*NFFT + idx_in] <= fft_in_I;
            in_mem_Q[wr_ptr*NFFT + idx_in] <= fft_in_Q;
            if (idx_in == (NFFT-1)) begin
                if (qcount < QDEPTH) begin
                    wr_ptr <= wrap_inc(wr_ptr);
                    qcount <= qcount + 1;
                end else begin
                    $display("[%0t] ERROR: input queue overflow", $time); $finish;
                end
                in_samp <= 0;
            end else in_samp <= idx_in + 1;
        end
    end

    always @(posedge clk_fast) begin
        if (rst) begin
            rd_ptr<=0; checking<=0; out_samp<=0;
            checked_frames<=0; frame_errs<=0; total_errs<=0;
        end else if (ifft_out_valid) begin
            idx_out = ifft_out_start ? 0 : out_samp;
            if (ifft_out_start) begin
                if (checking) begin
                    if (frame_errs==0)
                        $display("[IFFT][FRAME %0d] PASS", checked_frames-1);
                    else
                        $display("[IFFT][FRAME %0d] FAIL errs=%0d", checked_frames-1, frame_errs);
                end
                if (checked_frames < FRAMES_TO_CHECK) begin
                    if (qcount==0) begin checking<=0; end
                    else begin checking<=1; frame_errs<=0; checked_frames<=checked_frames+1; end
                end else checking<=0;
            end
            if (checking) begin
                dI = $signed(ifft_out_I) - $signed(in_mem_I[rd_ptr*NFFT + idx_out]);
                dQ = $signed(ifft_out_Q) - $signed(in_mem_Q[rd_ptr*NFFT + idx_out]);
                if (iabs(dI)>TOL || iabs(dQ)>TOL) begin
                    frame_errs<=frame_errs+1; total_errs<=total_errs+1;
                end
                if (idx_out==(NFFT-1)) begin
                    rd_ptr<=wrap_inc(rd_ptr); qcount<=qcount-1; out_samp<=0;
                end else out_samp<=idx_out+1;
            end else begin
                if (idx_out==(NFFT-1)) out_samp<=0; else out_samp<=idx_out+1;
            end
        end
    end

    // ============================================================
    // BLOQUE 7 — DISCARD_N Monitor
    //
    // Verificaciones:
    //   P1) Exactamente N_HALF=16 salidas válidas por frame
    //   P2) Exactamente 1 pulso dn_out_start por frame
    //   P4) dn_out[j] == ifft_out[N_HALF+j]  (discard_n no modifica el dato)
    //
    // Estrategia — delay line directa sobre ifft_out (sin for en always):
    //
    //   discard_n tiene 1 ciclo de latencia (registro de salida).
    //   Por tanto:  dn_out en ciclo T  ==  ifft_out en ciclo T-1
    //
    //   Se registra ifft_out un ciclo: ifft_d1_I/Q.
    //   Se registra también ifft_out_valid/start un ciclo: ifft_d1_valid/start.
    //
    //   Luego, en cada ciclo donde dn_out_valid=1, se compara:
    //     dn_out_I/Q  vs  ifft_d1_I/Q   (valor de 1 ciclo atrás)
    //
    //   Esta comparación es directa ciclo a ciclo, sin necesidad de
    //   ningún snapshot ni for-loop. La lógica de P1/P2 cuenta
    //   válidos y starts por frame usando dn_out_start como separador.
    //
    //   Condición de arranque: solo comparar cuando el delay line está
    //   caliente (dn_d1_armed=1, se activa al primer dn_out_valid).
    // ============================================================

    // Delay 1 ciclo de ifft_out (para alinear con dn_out)
    reg signed [WN-1:0] ifft_d1_I, ifft_d1_Q;
    reg                  ifft_d1_valid, ifft_d1_start;

    always @(posedge clk_fast) begin
        if (rst) begin
            ifft_d1_I     <= {WN{1'b0}};
            ifft_d1_Q     <= {WN{1'b0}};
            ifft_d1_valid <= 1'b0;
            ifft_d1_start <= 1'b0;
        end else begin
            ifft_d1_I     <= ifft_out_I;
            ifft_d1_Q     <= ifft_out_Q;
            ifft_d1_valid <= ifft_out_valid;
            ifft_d1_start <= ifft_out_start;
        end
    end

    // Contadores DISCARD_N
    integer dn_total_errs;
    integer dn_frames_checked;
    integer dn_frame_errs;
    integer dn_frame_valid_cnt;
    integer dn_frame_start_cnt;
    integer dn_p1_errs;
    integer dn_p2_errs;
    integer dn_p4_errs;
    integer dn_armed;       // 1 tras el primer dn_out_valid (delay caliente)

    integer dI_dn, dQ_dn;

    initial begin
        dn_total_errs=0; dn_frames_checked=0; dn_frame_errs=0;
        dn_frame_valid_cnt=0; dn_frame_start_cnt=0;
        dn_p1_errs=0; dn_p2_errs=0; dn_p4_errs=0;
        dn_armed=0;
    end

    always @(posedge clk_fast) begin : dn_monitor
        if (rst) begin
            dn_total_errs<=0; dn_frames_checked<=0; dn_frame_errs<=0;
            dn_frame_valid_cnt<=0; dn_frame_start_cnt<=0;
            dn_p1_errs<=0; dn_p2_errs<=0; dn_p4_errs<=0;
            dn_armed<=0;
        end else begin

            // Armar el delay tras el primer dn_out_valid
            if (dn_out_valid) dn_armed <= 1;

            // ---- Conteo, reset y comparación — un solo if/else para evitar
            //      conflicto de non-blocking en el ciclo de start.
            //
            //      Cuando dn_out_start=1 (= primer válido del nuevo frame):
            //        · Reportar frame anterior
            //        · Resetear contadores a 1 (ya cuenta esta muestra)
            //        · Comparar P4 normalmente
            //
            //      Cuando dn_out_valid=1 y NO es start:
            //        · Incrementar contadores
            //        · Comparar P4
            // ----
            if (dn_out_valid) begin

                if (dn_out_start) begin
                    // -- Reporte del frame anterior --
                    if (dn_frames_checked > 0) begin
                        if (dn_frame_valid_cnt !== N_HALF) begin
                            dn_p1_errs    <= dn_p1_errs + 1;
                            dn_total_errs <= dn_total_errs + 1;
                            $display("[DN][FRAME %0d] FAIL P1: valid_cnt=%0d exp=%0d",
                                dn_frames_checked-1, dn_frame_valid_cnt, N_HALF);
                        end
                        if (dn_frame_start_cnt !== 1) begin
                            dn_p2_errs    <= dn_p2_errs + 1;
                            dn_total_errs <= dn_total_errs + 1;
                            $display("[DN][FRAME %0d] FAIL P2: start_cnt=%0d exp=1",
                                dn_frames_checked-1, dn_frame_start_cnt);
                        end
                        if (dn_frame_errs == 0)
                            $display("[DN][FRAME %0d] PASS  valid=%0d  errs=0",
                                dn_frames_checked-1, dn_frame_valid_cnt);
                        else
                            $display("[DN][FRAME %0d] FAIL  valid=%0d  errs=%0d",
                                dn_frames_checked-1, dn_frame_valid_cnt, dn_frame_errs);
                    end
                    // -- Reset contadores: start ya cuenta como 1 válido y 1 start --
                    dn_frame_valid_cnt <= 1;
                    dn_frame_start_cnt <= 1;
                    dn_frame_errs      <= 0;
                    dn_frames_checked  <= dn_frames_checked + 1;

                end else begin
                    // -- Muestra normal: solo incrementar --
                    dn_frame_valid_cnt <= dn_frame_valid_cnt + 1;
                end

                // -- P4: comparar dato (aplica en start y en el resto) --
                if (dn_armed) begin
                    dI_dn = $signed(dn_out_I) - $signed(ifft_d1_I);
                    dQ_dn = $signed(dn_out_Q) - $signed(ifft_d1_Q);
                    if (iabs(dI_dn) > 0 || iabs(dQ_dn) > 0) begin
                        dn_frame_errs <= dn_frame_errs + 1;
                        dn_p4_errs    <= dn_p4_errs + 1;
                        dn_total_errs <= dn_total_errs + 1;
                        $display("[DN][FRAME %0d] FAIL P4: dn_I=%0d dn_Q=%0d ifft_d1_I=%0d ifft_d1_Q=%0d",
                            dn_frames_checked-1,
                            $signed(dn_out_I), $signed(dn_out_Q),
                            $signed(ifft_d1_I), $signed(ifft_d1_Q));
                    end
                end

            end
        end
    end

    // ============================================================
    // RESUMEN FINAL  — disparado cuando IFFT termina sus 30 frames
    // ============================================================
    always @(posedge clk_fast) begin
        if (!rst && ifft_out_valid && checked_frames == FRAMES_TO_CHECK
            && out_samp == (NFFT-1)) begin

            if (frame_errs==0)
                $display("[IFFT][FRAME %0d] PASS", checked_frames-1);
            else
                $display("[IFFT][FRAME %0d] FAIL errs=%0d", checked_frames-1, frame_errs);

            $display("");
            $display("========================================");
            $display("[TB] RESUMEN FINAL  v5 (con discard_n)");
            $display("========================================");
            $display("[OS]   %s  overflow=%0d  period_warns=%0d",
                (os_overflow||os_warns!=0) ? "FAIL":"PASS", os_overflow, os_warns);
            $display("[FIFO] %s  overflow=%0d  max_count=%0d (~%0d frames)",
                fifo_overflow?"FAIL":"PASS",
                fifo_overflow, fifo_max_count, (fifo_max_count+NFFT-1)/NFFT);
            $display("[FFT]  frames_out=%0d  period_warns=%0d  %s",
                fft_frames_out, fft_period_warns,
                fft_period_warns?"WARN":"OK");
            $display("[HB]   frames_checked=%0d  curr_errs=%0d  old_errs=%0d  => %s",
                hb_frame_total, hb_total_curr_errs, hb_total_old_errs,
                (hb_total_curr_errs==0&&hb_total_old_errs==0)?"PASS":"FAIL");
            $display("[CMUL] frames_checked=%0d  total_errs=%0d  misalign=%0d  => %s",
                cmul_frames_checked, cmul_total_errs, cmul_start_misalign,
                (cmul_total_errs==0&&cmul_start_misalign==0)?"PASS: Y=W*X=X (W=identidad)":"FAIL");
            $display("[IFFT] checked_frames=%0d  total_errs=%0d  tol=%0d  => %s",
                FRAMES_TO_CHECK, total_errs, TOL,
                total_errs==0?"PASS: IFFT(W*FFT(x))=x":"FAIL");
            $display("[DN]   frames_checked=%0d  total_errs=%0d  => %s",
                dn_frames_checked,
                dn_total_errs,
                dn_total_errs==0 ?
                    "PASS: dn_out=ifft[N..2N-1]" : "FAIL");
            $display("  [DN] P1(valid_cnt) errs=%0d  P2(start_cnt) errs=%0d  P4(datos) errs=%0d",
                dn_p1_errs, dn_p2_errs, dn_p4_errs);
            $display("========================================");
            $finish;
        end
    end

    // Timeout
    initial begin
        @(negedge rst);
        #3_000_000;
        $display("[TB] TIMEOUT  checked_frames=%0d  dn_frames=%0d",
            checked_frames, dn_frames_checked);
        $finish;
    end

endmodule

`default_nettype wire
