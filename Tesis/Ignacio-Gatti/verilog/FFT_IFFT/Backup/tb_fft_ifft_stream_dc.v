`timescale 1ns/1ps
`default_nettype none

module tb_fft_ifft_stream_dc;

  localparam integer NFFT = 32;
  localparam integer LOGN = 5;

  localparam integer NB_SRC  = 9;
  localparam integer NBF_SRC = 7;

  localparam integer NB_INT  = 17;
  localparam integer NBF_INT = 10;

  localparam integer NB_DST  = 9;
  localparam integer NBF_DST = 7;

  localparam integer TOL = 2;
  localparam integer DROP_FRAMES     = 1;
  localparam integer FRAMES_TO_CHECK = 10;

  // DC constant in Q7 => 1.0 = 1<<NBF_SRC
  localparam integer CONST_Q7 = (1 << NBF_SRC);

  reg clk, rst;
  initial begin clk=0; forever #5 clk=~clk; end
  initial begin rst=1; repeat(10) @(posedge clk); rst=0; end

  // source: 1 sample/clk, frame each 32 samples
  reg src_valid, src_start;
  reg signed [NB_SRC-1:0] src_xI, src_xQ;

  reg signed [NB_SRC-1:0] xI [0:NFFT-1];
  reg signed [NB_SRC-1:0] xQ [0:NFFT-1];
  integer n;

  // ===================== DC STIMULUS =====================
  initial begin
    for (n=0; n<NFFT; n=n+1) begin
      xI[n] = $signed(CONST_Q7); // constante 1.0
      xQ[n] = 0;
    end
  end

  integer sn;
  initial begin
    src_valid=0; src_start=0; src_xI=0; src_xQ=0; sn=0;
    @(negedge rst);
    src_valid=1;
    forever begin
      @(posedge clk);
      src_start <= (sn==0);
      src_xI    <= xI[sn];
      src_xQ    <= xQ[sn];
      if (sn==NFFT-1) sn <= 0;
      else            sn <= sn + 1;
    end
  end

  // DUTs
  wire fft_v, fft_s;
  wire signed [NB_INT-1:0] fft_I, fft_Q;

  wire ifft_v, ifft_s;
  wire signed [NB_DST-1:0] yI, yQ;

  wire fft_rdy, ifft_rdy;

  // REORDER_BITREV=1 en ambos para tener orden natural en espectro y en tiempo.
  fft_ifft_stream #(
    .NFFT(NFFT), .LOGN(LOGN),
    .NB_IN(NB_SRC),  .NBF_IN(NBF_SRC),
    .NB_W(NB_INT),   .NBF_W(NBF_INT),
    .NB_OUT(NB_INT), .NBF_OUT(NBF_INT),
    .SCALE_STAGE(0),
    .REORDER_BITREV(1)
  ) u_fft (
    .i_clk(clk), .i_rst(rst),
    .i_valid(src_valid), .i_start(src_start),
    .i_xI(src_xI), .i_xQ(src_xQ),
    .i_inverse(1'b0),
    .o_in_ready(fft_rdy),
    .o_start(fft_s), .o_valid(fft_v),
    .o_yI(fft_I), .o_yQ(fft_Q)
  );

  fft_ifft_stream #(
    .NFFT(NFFT), .LOGN(LOGN),
    .NB_IN(NB_INT), .NBF_IN(NBF_INT),
    .NB_W(NB_INT),  .NBF_W(NBF_INT),
    .NB_OUT(NB_DST),.NBF_OUT(NBF_DST),
    .SCALE_STAGE(0),
    .REORDER_BITREV(1)
  ) u_ifft (
    .i_clk(clk), .i_rst(rst),
    .i_valid(fft_v), .i_start(fft_s),
    .i_xI(fft_I), .i_xQ(fft_Q),
    .i_inverse(1'b1),
    .o_in_ready(ifft_rdy),
    .o_start(ifft_s), .o_valid(ifft_v),
    .o_yI(yI), .o_yQ(yQ)
  );

  // ------------------------------------------------------------
  // Helper
  // ------------------------------------------------------------
  function integer iabs;
    input integer v;
    begin iabs = (v < 0) ? -v : v; end
  endfunction

  // ============================================================
  // (1) FFT DC CHECKER: x[n]=const => X[0]=N*const, X[k!=0]~0
  // ============================================================
  localparam integer FFT_DROP_FRAMES = 1;  // ignora 1 frame de FFT por warm-up
  localparam integer FFT_TOL_DC      = 16; // tolerancia DC en LSB del formato NB_INT/NBF_INT
  localparam integer FFT_TOL_SPUR    = 16; // tolerancia spurs

  reg signed [NB_INT-1:0] X_I [0:NFFT-1];
  reg signed [NB_INT-1:0] X_Q [0:NFFT-1];

  integer fft_frame_seen_dc;
  integer fft_bin_dc;
  integer fft_done_dc;

    task check_fft_dc;
    integer k;
    integer max_spur;
    integer mag0;
    integer mag;
    integer exp_dc_norm;   // esperado si FFT normalizada (1/N): ~1.0
    integer exp_dc_un;     // esperado si FFT no normalizada: ~N*1.0
    integer const_int;
    integer d_norm, d_un;

    begin
      // const en Q7 -> interno Q10 (shift 3)
      const_int   = (CONST_Q7 <<< (NBF_INT - NBF_SRC));  // 1.0 en Q10 => 1024
      exp_dc_norm = const_int;           // FFT normalizada por 1/N
      exp_dc_un   = const_int * NFFT;    // FFT no normalizada

      // Magnitud DC (solo eje real para DC)
      mag0 = iabs($signed(X_I[0]));   // uso abs porque puede salir con signo global

      // Detectar cuál escala está usando realmente (norm vs un)
      d_norm = iabs(mag0 - exp_dc_norm);
      d_un   = iabs(mag0 - exp_dc_un);

      if (d_norm < d_un)
        $display("[FFT-DC] INFO: FFT parece NORMALIZADA (X0~1.0). |X0|=%0d exp=%0d", mag0, exp_dc_norm);
      else
        $display("[FFT-DC] INFO: FFT parece NO normalizada (X0~N). |X0|=%0d exp=%0d", mag0, exp_dc_un);

      // Max spur en bins k!=0 (magnitud aprox L1: |I| o |Q|)
      max_spur = 0;
      for (k=1; k<NFFT; k=k+1) begin
        mag = iabs($signed(X_I[k]));
        if (mag > max_spur) max_spur = mag;
        mag = iabs($signed(X_Q[k]));
        if (mag > max_spur) max_spur = mag;
      end

      $display("[FFT-DC] X0(I=%0d Q=%0d)  max_spur=%0d", $signed(X_I[0]), $signed(X_Q[0]), max_spur);

      // Criterio robusto: DC debe dominar fuerte.
      // Ejemplo: DC al menos 8x mayor que el mayor spur (≈ 18 dB)
      if (max_spur == 0) begin
        $display("[FFT-DC] PASS: spurs=0");
      end else if (mag0 >= (8*max_spur)) begin
        $display("[FFT-DC] PASS: DC domina (|X0|=%0d, max_spur=%0d, ratio~%0d:1)", mag0, max_spur, (mag0/max_spur));
      end else begin
        $display("[FFT-DC] FAIL: spurs demasiado grandes vs DC (|X0|=%0d, max_spur=%0d)", mag0, max_spur);
      end
      end
    endtask

  initial begin
    fft_frame_seen_dc = 0;
    fft_bin_dc        = 0;
    fft_done_dc       = 0;
  end

  always @(posedge clk) begin
    if (rst) begin
      fft_frame_seen_dc <= 0;
      fft_bin_dc        <= 0;
      fft_done_dc       <= 0;
    end else if (!fft_done_dc && fft_v) begin
      if (fft_s) begin
        fft_frame_seen_dc <= fft_frame_seen_dc + 1;
        fft_bin_dc        <= 0;
      end else begin
        fft_bin_dc <= fft_bin_dc + 1;
      end

      // capturo UN frame después del warm-up
      if (fft_frame_seen_dc == FFT_DROP_FRAMES) begin
        X_I[fft_bin_dc] = fft_I; // blocking a propósito (captura inmediata)
        X_Q[fft_bin_dc] = fft_Q;

        if (fft_bin_dc == NFFT-1) begin
          fft_done_dc = 1;
          check_fft_dc();
        end
      end
    end
  end

  // ============================================================
  // (2) Identidad IFFT(FFT(x))=x frame-by-frame + resumen final
  // ============================================================
  integer frame_seen;
  integer frame_chk;
  integer samp_idx;
  integer frame_errs;
  integer total_errs;
  integer prints;
  integer dI, dQ;

  initial begin
    frame_seen  = 0;
    frame_chk   = 0;
    samp_idx    = 0;
    frame_errs  = 0;
    total_errs  = 0;
    prints      = 0;

    @(negedge rst);
    #200000;
    $display("\n[TB] TIMEOUT total_errs=%0d checked_frames=%0d", total_errs, frame_chk);
    $finish;
  end

  always @(posedge clk) begin
    if (rst) begin
      frame_seen <= 0;
      frame_chk  <= 0;
      samp_idx   <= 0;
      frame_errs <= 0;
      total_errs <= 0;
      prints     <= 0;

    end else if (ifft_v) begin

      if (ifft_s) begin
        // reportar frame anterior (si ya estábamos chequeando)
        if (frame_seen > DROP_FRAMES) begin
          if (frame_errs == 0)
            $display("[FRAME %0d] PASS", frame_chk);
          else
            $display("[FRAME %0d] FAIL errs=%0d", frame_chk, frame_errs);
        end

        frame_seen <= frame_seen + 1;
        samp_idx   <= 0;
        frame_errs <= 0;

      end else begin
        samp_idx <= samp_idx + 1;
      end

      if (frame_seen >= DROP_FRAMES && frame_chk < FRAMES_TO_CHECK) begin
        dI = $signed(yI) - $signed(xI[samp_idx]);
        dQ = $signed(yQ) - $signed(xQ[samp_idx]);

        if (iabs(dI) > TOL || iabs(dQ) > TOL) begin
          frame_errs <= frame_errs + 1;
          total_errs <= total_errs + 1;

          if (prints < 10) begin
            $display("  mismatch frame=%0d samp=%0d inI=%0d outI=%0d dI=%0d",
                     frame_chk, samp_idx, $signed(xI[samp_idx]), $signed(yI), dI);
            prints <= prints + 1;
          end
        end

        if (samp_idx == NFFT-1) begin
          frame_chk <= frame_chk + 1;

          if (frame_chk + 1 == FRAMES_TO_CHECK) begin
            $display("\n[TB] DONE checked_frames=%0d total_errs=%0d (tol=%0d drop=%0d)",
                     FRAMES_TO_CHECK, total_errs, TOL, DROP_FRAMES);
            if (total_errs == 0) $display("[TB] PASS: IFFT(FFT(x))=x");
            else                 $display("[TB] FAIL");
            $finish;
          end
        end
      end
    end
  end

endmodule

`default_nettype wire