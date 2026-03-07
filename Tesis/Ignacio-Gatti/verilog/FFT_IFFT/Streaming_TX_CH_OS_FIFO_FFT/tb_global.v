`timescale 1ns/1ps
`default_nettype none

module tb_top_global_all;

  localparam integer WN      = 9;
  localparam integer NFFT    = 32;
  localparam integer FIFO_AW = 8;

  localparam integer TOL = 2;
  // Si querés que imprima hasta [FRAME 10], poné FRAMES_TO_CHECK = 11
  localparam integer FRAMES_TO_CHECK = 20;

  localparam integer QDEPTH = 64;
  localparam integer QMEM   = QDEPTH * NFFT;

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

  wire clk_low;

  wire signed [WN-1:0] tx_I_dbg, tx_Q_dbg, ch_I_dbg, ch_Q_dbg;
  wire os_overflow, os_start, os_valid;
  wire signed [WN-1:0] os_I, os_Q;

  wire fifo_full, fifo_empty, fifo_overflow;
  wire [FIFO_AW:0] fifo_count;

  wire fft_in_valid, fft_in_start;
  wire signed [WN-1:0] fft_in_I, fft_in_Q;

  wire fft_out_valid, fft_out_start;
  wire signed [16:0] fft_out_I, fft_out_Q;

  wire ifft_out_valid, ifft_out_start;
  wire signed [WN-1:0] ifft_out_I, ifft_out_Q;

  top_global_all #(
    .N_OS(16),
    .WN(WN),
    .FIFO_AW(FIFO_AW)
  ) dut (
    .clk_fast(clk_fast),
    .rst(rst),
    .enable_div(enable_div),
    .sigma_scale(sigma_scale),

    .clk_low(clk_low),

    .tx_I_dbg(tx_I_dbg),
    .tx_Q_dbg(tx_Q_dbg),
    .ch_I_dbg(ch_I_dbg),
    .ch_Q_dbg(ch_Q_dbg),

    .os_overflow(os_overflow),
    .os_start(os_start),
    .os_valid(os_valid),
    .os_I(os_I),
    .os_Q(os_Q),

    .fifo_full(fifo_full),
    .fifo_empty(fifo_empty),
    .fifo_overflow(fifo_overflow),
    .fifo_count(fifo_count),

    .fft_in_valid(fft_in_valid),
    .fft_in_start(fft_in_start),
    .fft_in_I(fft_in_I),
    .fft_in_Q(fft_in_Q),

    .fft_out_valid(fft_out_valid),
    .fft_out_start(fft_out_start),
    .fft_out_I(fft_out_I),
    .fft_out_Q(fft_out_Q),

    .ifft_out_valid(ifft_out_valid),
    .ifft_out_start(ifft_out_start),
    .ifft_out_I(ifft_out_I),
    .ifft_out_Q(ifft_out_Q)
  );

  function integer iabs;
    input integer v;
    begin
      if (v < 0) iabs = -v;
      else       iabs = v;
    end
  endfunction

  function integer wrap_inc;
    input integer ptr;
    begin
      if (ptr == (QDEPTH-1)) wrap_inc = 0;
      else                  wrap_inc = ptr + 1;
    end
  endfunction

  // ---------- OS checks ----------
  integer cyc;
  integer last_os_start_cyc;
  integer os_warns;

  initial begin
    cyc = 0;
    last_os_start_cyc = -1;
    os_warns = 0;
  end

  always @(posedge clk_fast) begin
    if (rst) begin
      cyc <= 0;
      last_os_start_cyc <= -1;
      os_warns <= 0;
    end else begin
      cyc <= cyc + 1;

      if (os_valid && os_start) begin
        if (last_os_start_cyc >= 0) begin
          if ((cyc - last_os_start_cyc) != NFFT) begin
            os_warns <= os_warns + 1;
            $display("[%0t] WARN: OS start period=%0d (esperado %0d)", $time, (cyc-last_os_start_cyc), NFFT);
          end
        end
        last_os_start_cyc <= cyc;

        $display("[%0t] FRAME_START  I=%0d Q=%0d  fifo_count=%0d", $time, $signed(os_I), $signed(os_Q), fifo_count);
      end
    end
  end

  // ---------- FIFO stats ----------
  integer fifo_max_count;
  always @(posedge clk_fast) begin
    if (rst) fifo_max_count <= 0;
    else if (fifo_count > fifo_max_count) fifo_max_count <= fifo_count;
  end

  // ---------- FFT/IFFT identity check ----------
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
      rd_ptr <= 0;
      checking <= 0;
      out_samp <= 0;
      checked_frames <= 0;
      frame_errs <= 0;
      total_errs <= 0;
    end else if (ifft_out_valid) begin

      idx_out = (ifft_out_start) ? 0 : out_samp;

      if (ifft_out_start) begin
        if (checking) begin
          if (frame_errs == 0) $display("[FRAME %0d] PASS", checked_frames-1);
          else                 $display("[FRAME %0d] FAIL  errs=%0d", checked_frames-1, frame_errs);
        end

        if (checked_frames < FRAMES_TO_CHECK) begin
          if (qcount == 0) begin
            $display("[%0t] WARN: salida IFFT frame pero cola vacía (subí QDEPTH)", $time);
            checking <= 0;
          end else begin
            checking <= 1;
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

          if (checked_frames == FRAMES_TO_CHECK) begin
            if (frame_errs == 0) $display("[FRAME %0d] PASS", checked_frames-1);
            else                 $display("[FRAME %0d] FAIL  errs=%0d", checked_frames-1, frame_errs);

            $display("\n[TB] DONE checked_frames=%0d total_errs=%0d (tol=%0d)",
                     FRAMES_TO_CHECK, total_errs, TOL);

            if (os_overflow || os_warns != 0)
              $display("[OS]   FAIL  overflow=%0d warns=%0d", os_overflow, os_warns);
            else
              $display("[OS]   PASS  overflow=0 period_ok");

            if (fifo_overflow)
              $display("[FIFO] FAIL  overflow=1 max_count=%0d", fifo_max_count);
            else
              $display("[FIFO] PASS  overflow=0 max_count=%0d (~%0d frames)",
                       fifo_max_count, (fifo_max_count+NFFT-1)/NFFT);

            if (total_errs == 0)
              $display("[FFT]  PASS: IFFT(FFT(x))=x");
            else
              $display("[FFT]  FAIL");

            $finish;
          end
        end else begin
          out_samp <= idx_out + 1;
        end
      end else begin
        if (idx_out == (NFFT-1)) out_samp <= 0;
        else                    out_samp <= idx_out + 1;
      end

    end
  end

  initial begin
    @(negedge rst);
    #2000000;
    $display("\n[TB] TIMEOUT  checked_frames=%0d total_errs=%0d", checked_frames, total_errs);
    $finish;
  end

endmodule

`default_nettype wire