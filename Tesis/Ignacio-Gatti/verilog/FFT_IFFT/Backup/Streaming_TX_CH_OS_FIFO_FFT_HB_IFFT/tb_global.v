`timescale 1ns/1ps
`default_nettype none

// ============================================================
// tb_top_global_all  v4  (con cmul_pbfdaf, K=1, LMS-ready)
//
// Verificación por el Tcl Console de:
//   [FIFO]  estado y ocupación
//   [FFT]   frames procesados y start period
//   [HB]    banco activo, muestra idx, coherencia curr/old
//   [CMUL]  Y = W0*X_curr + W1*X_old  con W0=1+0j,W1=0 => Y == X_curr
//   [IFFT]  identidad IFFT(W*FFT(x)) = x  (tolerancia TOL)
//
// Cadena verificada:
//   FFT -> HB -> CMUL_PBFDAF(W0=identidad,W1=0) -> IFFT
//
// Estrategia de chequeo del CMUL:
//   Con W=identidad, la salida del CMUL debe ser igual a
//   la entrada (X_curr del HB), con 1 ciclo de latencia.
//   Se captura hb_out_curr un ciclo antes y se compara
//   con cmul_out en el ciclo siguiente.
//
// Latencias acumuladas desde fft_out:
//   HB   : +1 ciclo  -> hb_out_valid
//   CMUL : +1 ciclo  -> cmul_out_valid
//   Total: +2 ciclos desde fft_out hasta cmul_out
// ============================================================

module tb_top_global_all;

    // ============================================================
    // Parámetros del sistema
    // ============================================================
    localparam integer WN        = 9;
    localparam integer NFFT      = 32;
    localparam integer NB_INT    = 17;
    localparam integer FIFO_AW   = 8;
    localparam integer K_HIST    = 1;   // mismo que en top

    // Tolerancia para comparación (cuantización + escala IFFT)
    localparam integer TOL            = 3;
    localparam integer FRAMES_TO_CHECK = 30;  // frames que chequeamos de IFFT

    // Profundidad de la cola de captura (frames)
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
    initial sigma_scale = 11'd0;   // sin ruido para verificar identidad

    // ============================================================
    // Señales del DUT
    // ============================================================
    wire clk_low;
    wire signed [WN-1:0]    tx_I_dbg,  tx_Q_dbg;
    wire signed [WN-1:0]    ch_I_dbg,  ch_Q_dbg;

    wire                    os_overflow, os_start, os_valid;
    wire signed [WN-1:0]    os_I, os_Q;

    wire                    fifo_full, fifo_empty, fifo_overflow;
    wire [FIFO_AW:0]        fifo_count;

    wire                    fft_in_valid, fft_in_start;
    wire signed [WN-1:0]    fft_in_I, fft_in_Q;

    wire                    fft_out_valid, fft_out_start;
    wire signed [NB_INT-1:0] fft_out_I, fft_out_Q;

    wire                    hb_out_valid, hb_out_start;
    wire signed [NB_INT-1:0] hb_out_curr_I, hb_out_curr_Q;
    wire signed [NB_INT-1:0] hb_out_old_I,  hb_out_old_Q;

    // --- CMUL ---
    wire                    cmul_out_valid, cmul_out_start;
    wire signed [NB_INT-1:0] cmul_out_I, cmul_out_Q;

    wire                    ifft_out_valid, ifft_out_start;
    wire signed [WN-1:0]    ifft_out_I, ifft_out_Q;

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
        .ifft_out_Q    (ifft_out_Q)
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
        begin
            wrap_inc = (ptr == (QDEPTH-1)) ? 0 : (ptr + 1);
        end
    endfunction

    // ============================================================
    // ---- BLOQUE 1: OS Monitor ----
    //  Verifica que el periodo entre os_start sea NFFT ciclos.
    // ============================================================
    integer cyc;
    integer last_os_start_cyc;
    integer os_warns;

    initial begin
        cyc = 0; last_os_start_cyc = -1; os_warns = 0;
    end

    always @(posedge clk_fast) begin
        if (rst) begin
            cyc <= 0; last_os_start_cyc <= -1; os_warns <= 0;
        end else begin
            cyc <= cyc + 1;
            if (os_valid && os_start) begin
                if (last_os_start_cyc >= 0) begin
                    if ((cyc - last_os_start_cyc) != NFFT) begin
                        os_warns <= os_warns + 1;
                        $display("[%0t] WARN OS: periodo=%0d (esperado %0d)",
                                 $time, (cyc-last_os_start_cyc), NFFT);
                    end
                end
                last_os_start_cyc <= cyc;
            end
        end
    end

    // ============================================================
    // ---- BLOQUE 2: FIFO Monitor ----
    //  Registra ocupación máxima y detecta overflow.
    // ============================================================
    integer fifo_max_count;
    integer fifo_frame_writes;   // veces que os_valid && os_start

    initial begin fifo_max_count = 0; fifo_frame_writes = 0; end

    always @(posedge clk_fast) begin
        if (rst) begin
            fifo_max_count   <= 0;
            fifo_frame_writes <= 0;
        end else begin
            if ($signed(fifo_count) > fifo_max_count)
                fifo_max_count <= fifo_count;
            if (os_valid && os_start)
                fifo_frame_writes <= fifo_frame_writes + 1;
        end
    end

    // ============================================================
    // ---- BLOQUE 3: FFT Monitor ----
    //  Cuenta frames procesados y verifica start period.
    // ============================================================
    integer fft_frames_out;
    integer last_fft_start_cyc;
    integer fft_period_warns;

    initial begin
        fft_frames_out = 0; last_fft_start_cyc = -1; fft_period_warns = 0;
    end

    always @(posedge clk_fast) begin
        if (rst) begin
            fft_frames_out   <= 0;
            last_fft_start_cyc <= -1;
            fft_period_warns <= 0;
        end else if (fft_out_valid) begin
            if (fft_out_start) begin
                fft_frames_out <= fft_frames_out + 1;
                if (last_fft_start_cyc >= 0) begin
                    if ((cyc - last_fft_start_cyc) != NFFT) begin
                        fft_period_warns <= fft_period_warns + 1;
                        $display("[%0t] WARN FFT: periodo=%0d (esperado %0d)",
                                 $time, (cyc-last_fft_start_cyc), NFFT);
                    end
                end
                last_fft_start_cyc <= cyc;
            end
        end
    end

    // ============================================================
    // ---- BLOQUE 4: History Buffer Monitor (delay-line) ----
    //
    //  hb_out_curr[t] == fft_out[t-1]           (latencia 1 ciclo)
    //  hb_out_old[t]  == fft_out[t-1-K*NFFT]   (latencia 1 + K*NFFT ciclos)
    //
    //  Shift register de K*NFFT+1 entradas sobre fft_out_valid:
    //    dly[0]       = fft_out del ciclo valido inmediatamente anterior
    //    dly[K*NFFT]  = fft_out de K*NFFT ciclos validos atras
    //
    //  Comparacion ciclo-a-ciclo: sin contadores de frame, sin indexacion.
    // ============================================================

    // ---- Variables para BLOQUE 5 (deben declararse antes de BLOQUE 5) ----
    localparam integer DLY_DEPTH = K_HIST * NFFT + 1;

    reg signed [NB_INT-1:0] fft_dly_I [0:DLY_DEPTH-1];
    reg signed [NB_INT-1:0] fft_dly_Q [0:DLY_DEPTH-1];
    integer fft_dly_fill;   // cuantas entradas validas hay en el delay line

    // Cola FFT para BLOQUE 5
    reg signed [NB_INT-1:0] fft_q_I [0:QMEM-1];
    reg signed [NB_INT-1:0] fft_q_Q [0:QMEM-1];
    integer fft_wr_ptr;
    integer fft_q_count;
    integer fft_in_samp;

    // Contadores de error HB
    integer hb_total_curr_errs;
    integer hb_total_old_errs;
    integer hb_frame_errs_curr;
    integer hb_frame_errs_old;
    integer hb_frame_total;

    // Variables de delay (para BLOQUE 5)
    reg signed [NB_INT-1:0] fft_prev_I,  fft_prev2_I;
    reg signed [NB_INT-1:0] fft_prev_Q,  fft_prev2_Q;
    reg                      fft_prev_valid, fft_prev2_valid;

    integer dly_init_i;
    initial begin
        for (dly_init_i=0; dly_init_i<DLY_DEPTH; dly_init_i=dly_init_i+1) begin
            fft_dly_I[dly_init_i] = 0;
            fft_dly_Q[dly_init_i] = 0;
        end
        fft_dly_fill = 0;
        fft_wr_ptr = 0; fft_q_count = 0; fft_in_samp = 0;
        hb_total_curr_errs = 0; hb_total_old_errs = 0;
        hb_frame_errs_curr = 0; hb_frame_errs_old = 0;
        hb_frame_total = 0;
        fft_prev_valid = 0; fft_prev_I = 0; fft_prev_Q = 0;
        fft_prev2_valid = 0; fft_prev2_I = 0; fft_prev2_Q = 0;
    end

    // ---- Always A: desplazar delay line con cada muestra FFT valida ----
    integer sh_i;
    always @(posedge clk_fast) begin : hb_dly
        integer qi_wr;
        if (rst) begin
            fft_dly_fill   <= 0;
            fft_wr_ptr     <= 0;
            fft_q_count    <= 0;
            fft_in_samp    <= 0;
            fft_prev_valid <= 0; fft_prev2_valid <= 0;
        end else begin
            fft_prev_valid  <= fft_out_valid;
            fft_prev_I      <= fft_out_I;
            fft_prev_Q      <= fft_out_Q;
            fft_prev2_valid <= fft_prev_valid;
            fft_prev2_I     <= fft_prev_I;
            fft_prev2_Q     <= fft_prev_Q;

            if (fft_out_valid) begin
                // Shift
                for (sh_i = DLY_DEPTH-1; sh_i > 0; sh_i = sh_i-1) begin
                    fft_dly_I[sh_i] <= fft_dly_I[sh_i-1];
                    fft_dly_Q[sh_i] <= fft_dly_Q[sh_i-1];
                end
                fft_dly_I[0] <= fft_out_I;
                fft_dly_Q[0] <= fft_out_Q;
                if (fft_dly_fill < DLY_DEPTH)
                    fft_dly_fill <= fft_dly_fill + 1;

                // Cola para BLOQUE 5
                if (fft_out_start) fft_in_samp <= 0;
                qi_wr = (fft_wr_ptr % QDEPTH) * NFFT + fft_in_samp;
                fft_q_I[qi_wr] <= fft_out_I;
                fft_q_Q[qi_wr] <= fft_out_Q;
                if (fft_in_samp == NFFT-1) begin
                    fft_wr_ptr  <= wrap_inc(fft_wr_ptr);
                    fft_q_count <= (fft_q_count < QDEPTH) ? fft_q_count+1 : fft_q_count;
                    fft_in_samp <= 0;
                end else
                    fft_in_samp <= fft_in_samp + 1;
            end
        end
    end

    // ---- Always B: comparar hb_out con delay line ----
    // Timing exacto:
    //   fft_out_valid ciclo T  -> dly shift ocurre en flanco T
    //                          -> dly[0] = fft_out[T] disponible DESPUES del flanco T
    //   hb_out_valid  ciclo T+1 (1 ciclo de latencia del HB)
    //                          -> hb_out_curr[T+1] = fft_out[T] = dly[0] post-flanco T
    //   En el flanco T+1: dly[0] ya tiene fft_out[T] (de la escritura en flanco T)
    //   -> comparar hb_out_curr con dly[0] ES CORRECTO en el flanco T+1
    //
    //   hb_out_old[T+1] = fft_out[T - K*NFFT]
    //   En flanco T+1: dly[K*NFFT] tiene fft_out[T - K*NFFT] (K*NFFT shifts atras)
    //   -> comparar hb_out_old con dly[K*NFFT] ES CORRECTO en el flanco T+1
    always @(posedge clk_fast) begin : hb_chk
        integer dIc3, dQc3, dIo4, dQo4;
        if (rst) begin
            hb_total_curr_errs <= 0; hb_total_old_errs  <= 0;
            hb_frame_errs_curr <= 0; hb_frame_errs_old  <= 0;
            hb_frame_total     <= 0;
        end else if (hb_out_valid) begin

            // Reporte de frame anterior en cada start
            if (hb_out_start) begin
                if (hb_frame_total > 0) begin
                    if (hb_frame_errs_curr==0 && hb_frame_errs_old==0)
                        $display("[HB][FRAME %0d] PASS  curr_errs=0  old_errs=0",
                                  hb_frame_total-1);
                    else
                        $display("[HB][FRAME %0d] FAIL  curr_errs=%0d  old_errs=%0d",
                                  hb_frame_total-1, hb_frame_errs_curr, hb_frame_errs_old);
                end
                hb_frame_errs_curr <= 0;
                hb_frame_errs_old  <= 0;
                hb_frame_total     <= hb_frame_total + 1;
            end else begin
                // Comparar X_curr con dly[0]  (solo !start porque start=latch de la muestra 0,
                // ya verificada: en el ciclo start hb_out_curr = muestra-0 del frame N,
                // y dly[0] = muestra-31 del frame N-1... distinto frame -> no comparar en start)
                dIc3 = $signed(hb_out_curr_I) - $signed(fft_dly_I[0]);
                dQc3 = $signed(hb_out_curr_Q) - $signed(fft_dly_Q[0]);
                if (iabs(dIc3) > TOL || iabs(dQc3) > TOL) begin
                    hb_frame_errs_curr <= hb_frame_errs_curr + 1;
                    hb_total_curr_errs <= hb_total_curr_errs + 1;
                end

                // Comparar X_old con dly[K_HIST*NFFT] (solo cuando el delay esta lleno)
                if (fft_dly_fill >= DLY_DEPTH) begin
                    dIo4 = $signed(hb_out_old_I) - $signed(fft_dly_I[K_HIST*NFFT]);
                    dQo4 = $signed(hb_out_old_Q) - $signed(fft_dly_Q[K_HIST*NFFT]);
                    // Debug primeras 3 muestras del frame 1
                    if (hb_frame_total == 1 && hb_frame_errs_old < 3)
                        $display("[DBG3] s=%0d old_I=%0d dly[%0d]_I=%0d diff=%0d",
                            hb_frame_errs_old+hb_frame_errs_curr,
                            $signed(hb_out_old_I), K_HIST*NFFT,
                            $signed(fft_dly_I[K_HIST*NFFT]), dIo4);
                    if (iabs(dIo4) > TOL || iabs(dQo4) > TOL) begin
                        hb_frame_errs_old  <= hb_frame_errs_old  + 1;
                        hb_total_old_errs  <= hb_total_old_errs  + 1;
                    end
                end
            end
        end
    end

    // ============================================================
    // ---- BLOQUE 5: CMUL Monitor ----
    //
    //  Con W = identidad (1+0j): cmul_out debe ser igual a hb_out_curr
    //  con exactamente 1 ciclo de latencia.
    //
    //  Estrategia:
    //    - Registrar hb_out_curr un ciclo (hb_prev_*) 
    //    - En el ciclo siguiente, cuando cmul_out_valid=1,
    //      comparar cmul_out con hb_prev
    //    - Tolerancia TOL (misma que IFFT): el complex_mult tiene
    //      sat_trunc round-to-even, puede haber 1 LSB de diferencia.
    //
    //  Nota: el start del CMUL se verifica por separado para
    //  confirmar que la alineación de frames no se rompió.
    // ============================================================
    reg signed [NB_INT-1:0] hb_prev_I, hb_prev_Q;
    reg                      hb_prev_valid, hb_prev_start;

    integer cmul_total_errs;
    integer cmul_frames_checked;
    integer cmul_frame_errs;
    integer cmul_start_misalign;

    integer dIcm, dQcm;

    initial begin
        hb_prev_I       = 0; hb_prev_Q       = 0;
        hb_prev_valid   = 0; hb_prev_start   = 0;
        cmul_total_errs     = 0;
        cmul_frames_checked = 0;
        cmul_frame_errs     = 0;
        cmul_start_misalign = 0;
    end

    always @(posedge clk_fast) begin : cmul_monitor
        if (rst) begin
            hb_prev_I           <= 0; hb_prev_Q       <= 0;
            hb_prev_valid       <= 0; hb_prev_start   <= 0;
            cmul_total_errs     <= 0;
            cmul_frames_checked <= 0;
            cmul_frame_errs     <= 0;
            cmul_start_misalign <= 0;
        end else begin
            // Registrar hb_out un ciclo hacia adelante
            hb_prev_valid <= hb_out_valid;
            hb_prev_start <= hb_out_start;
            hb_prev_I     <= hb_out_curr_I;
            hb_prev_Q     <= hb_out_curr_Q;

            // Cuando el CMUL produce salida, comparar con el HB del ciclo anterior
            if (cmul_out_valid) begin

                // Chequeo de alineación de start
                if (cmul_out_start && !hb_prev_start) begin
                    cmul_start_misalign <= cmul_start_misalign + 1;
                    $display("[CMUL][WARN] start desalineado en frame %0d",
                              cmul_frames_checked);
                end

                // Reporte de frame anterior al detectar nuevo start
                if (cmul_out_start) begin
                    if (cmul_frames_checked > 0) begin
                        if (cmul_frame_errs == 0)
                            $display("[CMUL][FRAME %0d] PASS  errs=0",
                                      cmul_frames_checked-1);
                        else
                            $display("[CMUL][FRAME %0d] FAIL  errs=%0d",
                                      cmul_frames_checked-1, cmul_frame_errs);
                    end
                    cmul_frame_errs     <= 0;
                    cmul_frames_checked <= cmul_frames_checked + 1;
                end

                // Comparar: con W=identidad, cmul_out == hb_prev (dentro de TOL)
                if (hb_prev_valid) begin
                    dIcm = $signed(cmul_out_I) - $signed(hb_prev_I);
                    dQcm = $signed(cmul_out_Q) - $signed(hb_prev_Q);
                    if (iabs(dIcm) > TOL || iabs(dQcm) > TOL) begin
                        cmul_frame_errs <= cmul_frame_errs + 1;
                        cmul_total_errs <= cmul_total_errs + 1;
                        $display("[CMUL][ERR] frame=%0d dI=%0d dQ=%0d",
                                  cmul_frames_checked-1, dIcm, dQcm);
                    end
                end
            end
        end
    end

    // ============================================================
    // ---- BLOQUE 6: IFFT identidad check ----
    //  Compara IFFT(FFT(x)) con la entrada original.
    //  Se usa la cola de ENTRADA a la FFT (fft_in_I/Q) para cotejar.
    //  Nota: ahora hay 1 ciclo extra de latencia del HB, por lo que
    //  la cola debe ser suficientemente grande.
    // ============================================================

    // Cola de frames de entrada FFT (para comparar con salida IFFT)
    reg signed [WN-1:0] in_mem_I [0:QMEM-1];
    reg signed [WN-1:0] in_mem_Q [0:QMEM-1];

    integer wr_ptr, rd_ptr, qcount;
    integer in_samp;
    integer checking, out_samp;
    integer checked_frames, frame_errs, total_errs;
    integer idx_in, idx_out;
    integer dI, dQ;

    initial begin
        wr_ptr = 0; rd_ptr = 0; qcount = 0;
        in_samp = 0;
        checking = 0; out_samp = 0;
        checked_frames = 0; frame_errs = 0; total_errs = 0;
    end

    // Captura entrada FFT
    always @(posedge clk_fast) begin
        if (rst) begin
            wr_ptr  <= 0;
            qcount  <= 0;
            in_samp <= 0;
        end else if (fft_in_valid) begin
            idx_in = (fft_in_start) ? 0 : in_samp;

            in_mem_I[wr_ptr*NFFT + idx_in] <= fft_in_I;
            in_mem_Q[wr_ptr*NFFT + idx_in] <= fft_in_Q;

            if (idx_in == (NFFT-1)) begin
                if (qcount < QDEPTH) begin
                    wr_ptr <= wrap_inc(wr_ptr);
                    qcount <= qcount + 1;
                end else begin
                    $display("[%0t] ERROR: input frame queue overflow (QDEPTH=%0d)", $time, QDEPTH);
                    $finish;
                end
                in_samp <= 0;
            end else begin
                in_samp <= idx_in + 1;
            end
        end
    end

    // Comparación salida IFFT
    always @(posedge clk_fast) begin
        if (rst) begin
            rd_ptr         <= 0;
            checking       <= 0;
            out_samp       <= 0;
            checked_frames <= 0;
            frame_errs     <= 0;
            total_errs     <= 0;
        end else if (ifft_out_valid) begin

            idx_out = (ifft_out_start) ? 0 : out_samp;

            if (ifft_out_start) begin
                // Reporte del frame anterior
                if (checking) begin
                    if (frame_errs == 0)
                        $display("[IFFT][FRAME %0d] PASS", checked_frames-1);
                    else
                        $display("[IFFT][FRAME %0d] FAIL  errs=%0d", checked_frames-1, frame_errs);
                end

                if (checked_frames < FRAMES_TO_CHECK) begin
                    if (qcount == 0) begin
                        $display("[%0t] WARN IFFT: frame salida pero cola vacía", $time);
                        checking <= 0;
                    end else begin
                        checking   <= 1;
                        frame_errs <= 0;
                        checked_frames <= checked_frames + 1;
                    end
                end else begin
                    checking <= 0;
                end
            end

            if (checking) begin
                dI = $signed(ifft_out_I) - $signed(in_mem_I[rd_ptr*NFFT + idx_out]);
                dQ = $signed(ifft_out_Q) - $signed(in_mem_Q[rd_ptr*NFFT + idx_out]);

                if (iabs(dI) > TOL || iabs(dQ) > TOL) begin
                    frame_errs <= frame_errs + 1;
                    total_errs <= total_errs + 1;
                end

                if (idx_out == (NFFT-1)) begin
                    rd_ptr <= wrap_inc(rd_ptr);
                    qcount <= qcount - 1;
                    out_samp <= 0;

                    // ---- RESULTADO FINAL ----
                    if (checked_frames == FRAMES_TO_CHECK) begin
                        if (frame_errs == 0)
                            $display("[IFFT][FRAME %0d] PASS", checked_frames-1);
                        else
                            $display("[IFFT][FRAME %0d] FAIL  errs=%0d", checked_frames-1, frame_errs);

                        $display("");
                        $display("========================================");
                        $display("[TB] RESUMEN FINAL");
                        $display("========================================");
                        $display("[OS]   %s  overflow=%0d  period_warns=%0d",
                                 (os_overflow || os_warns != 0) ? "FAIL" : "PASS",
                                 os_overflow, os_warns);

                        $display("[FIFO] %s  overflow=%0d  max_count=%0d (~%0d frames)",
                                 fifo_overflow ? "FAIL" : "PASS",
                                 fifo_overflow, fifo_max_count,
                                 (fifo_max_count + NFFT - 1) / NFFT);

                        $display("[FFT]  frames_out=%0d  period_warns=%0d  %s",
                                 fft_frames_out, fft_period_warns,
                                 fft_period_warns ? "WARN" : "OK");

                        $display("[HB]   frames_checked=%0d  curr_errs=%0d  old_errs=%0d  => %s",
                                 hb_frame_total,
                                 hb_total_curr_errs, hb_total_old_errs,
                                 (hb_total_curr_errs==0 && hb_total_old_errs==0) ? "PASS" : "FAIL");

                        $display("[CMUL] frames_checked=%0d  total_errs=%0d  misalign=%0d  => %s",
                                 cmul_frames_checked, cmul_total_errs, cmul_start_misalign,
                                 (cmul_total_errs==0 && cmul_start_misalign==0) ? "PASS: Y=W*X=X (W=identidad)" : "FAIL");

                        $display("[IFFT] checked_frames=%0d  total_errs=%0d  tol=%0d  => %s",
                                 FRAMES_TO_CHECK, total_errs, TOL,
                                 (total_errs == 0) ? "PASS: IFFT(W*FFT(x))=x" : "FAIL");

                        $display("========================================");
                        $finish;
                    end
                end else begin
                    out_samp <= idx_out + 1;
                end
            end else begin
                if (idx_out == (NFFT-1)) out_samp <= 0;
                else                     out_samp <= idx_out + 1;
            end
        end
    end

    // ============================================================
    // Timeout de seguridad
    // ============================================================
    initial begin
        @(negedge rst);
        #3000000;
        $display("[TB] TIMEOUT  checked_frames=%0d  total_errs=%0d  hb_checked=%0d",
                 checked_frames, total_errs, hb_frame_total);
        $finish;
    end

endmodule

`default_nettype wire
