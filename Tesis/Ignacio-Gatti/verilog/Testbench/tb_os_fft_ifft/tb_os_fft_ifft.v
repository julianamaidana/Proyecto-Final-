`timescale 1ns/1ps

module tb_os_fft_ifft;

  // ============================================================
  // PARAMETROS
  // ============================================================
  localparam integer N       = 16;     // os_buffer collect
  localparam integer NFFT    = 32;     // FFT size (=2N)
  localparam integer LOGN    = 5;      // log2(32)

  localparam integer NB_IN   = 9;      // Q(9,7)
  localparam integer NBF_IN  = 7;

  localparam integer NB_W    = 17;     // Q(17,10)
  localparam integer NBF_W   = 10;

  localparam integer NB_OUT  = 9;      // Q(9,7)
  localparam integer NBF_OUT = 7;

  localparam integer NFRAMES     = 100;     // stress frames
  localparam integer TIMEOUT_CYC = 400000;  // timeout cycles
  localparam integer TOL_LSB     = 1;       // tolerancia +/-1 LSB

  // probabilidad de burbuja al enviar input (0..100)
  localparam integer BUBBLE_PCT  = 35;
  // threshold 0..255 aproximado
  localparam integer BUBBLE_THR  = (BUBBLE_PCT * 256) / 100;

  // ============================================================
  // CLOCK / RESET
  // ============================================================
  reg i_clk, i_rst;

  initial i_clk = 1'b0;
  always #5 i_clk = ~i_clk;

  // ============================================================
  // I/O chain
  // ============================================================
  // input to os_buffer
  reg                    i_valid;
  reg signed [NB_IN-1:0] i_xI, i_xQ;
  wire                   os_in_ready;

  // os_buffer -> FFT
  wire                    os_fft_start;
  wire                    os_fft_valid;
  wire signed [NB_IN-1:0] os_fft_xI, os_fft_xQ;

  // FFT -> IFFT
  wire                   fft_in_ready;
  wire                   fft_start;
  wire                   fft_valid;
  wire signed [NB_W-1:0] fft_yI_w;
  wire signed [NB_W-1:0] fft_yQ_w;

  // IFFT output
  wire                    ifft_in_ready;
  wire                    ifft_start;
  wire                    ifft_valid;
  wire signed [NB_OUT-1:0] ifft_yI;
  wire signed [NB_OUT-1:0] ifft_yQ;

  // ============================================================
  // DUTs
  // ============================================================

  os_buffer #(
    .N(N),
    .WN(NB_IN)
  ) u_os (
    .i_clk(i_clk),
    .i_rst(i_rst),
    .i_valid(i_valid),
    .i_xI(i_xI),
    .i_xQ(i_xQ),
    .o_in_ready(os_in_ready),
    .o_fft_start(os_fft_start),
    .o_fft_valid(os_fft_valid),
    .o_fft_xI(os_fft_xI),
    .o_fft_xQ(os_fft_xQ)
  );

  // FFT: input narrow, output wide (NB_OUT=NB_W)
  fft_ifft #(
    .NFFT(NFFT),
    .LOGN(LOGN),
    .NB_IN(NB_IN),
    .NBF_IN(NBF_IN),
    .NB_W(NB_W),
    .NBF_W(NBF_W),
    .NB_OUT(NB_W),
    .NBF_OUT(NBF_W),
    .SCALE_STAGE(0)
  ) u_fft (
    .i_clk(i_clk),
    .i_rst(i_rst),
    .i_valid(os_fft_valid),
    .i_xI(os_fft_xI),
    .i_xQ(os_fft_xQ),
    .i_inverse(1'b0),
    .o_in_ready(fft_in_ready),
    .o_start(fft_start),
    .o_valid(fft_valid),
    .o_yI(fft_yI_w),
    .o_yQ(fft_yQ_w)
  );

  // IFFT: input wide, output narrow (NB_OUT=9)
  fft_ifft #(
    .NFFT(NFFT),
    .LOGN(LOGN),
    .NB_IN(NB_W),
    .NBF_IN(NBF_W),
    .NB_W(NB_W),
    .NBF_W(NBF_W),
    .NB_OUT(NB_OUT),
    .NBF_OUT(NBF_OUT),
    .SCALE_STAGE(0)
  ) u_ifft (
    .i_clk(i_clk),
    .i_rst(i_rst),
    .i_valid(fft_valid),
    .i_xI(fft_yI_w),
    .i_xQ(fft_yQ_w),
    .i_inverse(1'b1),
    .o_in_ready(ifft_in_ready),
    .o_start(ifft_start),
    .o_valid(ifft_valid),
    .o_yI(ifft_yI),
    .o_yQ(ifft_yQ)
  );

  // ============================================================
  // SCOREBOARD MEMORIES (esperado a la salida de IFFT)
  // ============================================================
  reg signed [NB_IN-1:0] prevI [0:N-1];
  reg signed [NB_IN-1:0] prevQ [0:N-1];
  reg signed [NB_IN-1:0] currI [0:N-1];
  reg signed [NB_IN-1:0] currQ [0:N-1];
  reg signed [NB_IN-1:0] expI  [0:NFFT-1];
  reg signed [NB_IN-1:0] expQ  [0:NFFT-1];

  integer k;
  integer err_cnt;

  // ============================================================
  // RNG: LFSR 32-bit (reproducible)
  // ============================================================
  reg [31:0] rng;

  function [31:0] lfsr_next;
    input [31:0] s;
    reg fb;
    begin
      fb = s[31] ^ s[21] ^ s[1] ^ s[0];
      lfsr_next = {s[30:0], fb};
    end
  endfunction

  // ============================================================
  // Helpers
  // ============================================================
  function integer iabs;
    input integer x;
    begin
      if (x < 0) iabs = -x;
      else       iabs = x;
    end
  endfunction

  function signed [NB_IN-1:0] clip_in;
    input integer x;
    integer maxv, minv;
    integer y;
    begin
      maxv = (1 << (NB_IN-1)) - 1;
      minv = -(1 << (NB_IN-1));
      y = x;
      if (y > maxv) y = maxv;
      if (y < minv) y = minv;
      clip_in = y[NB_IN-1:0];
    end
  endfunction

  // ============================================================
  // TASKS
  // ============================================================
  task init_prev_zero;
    begin
      for (k = 0; k < N; k = k + 1) begin
        prevI[k] = 0;
        prevQ[k] = 0;
      end
    end
  endtask

  task make_curr_block_random;
    input integer frame_id;
    integer v1, v2;
    begin
      for (k = 0; k < N; k = k + 1) begin
        rng = lfsr_next(rng);
        v1  = $signed(rng[15:0]);
        rng = lfsr_next(rng);
        v2  = $signed(rng[15:0]);

        // compresion + drift
        currI[k] = clip_in( (v1 >>> 8) + frame_id + k );
        currQ[k] = clip_in( (v2 >>> 8) - frame_id - 2*k );
      end
    end
  endtask

  task build_expected_frame;
    begin
      for (k = 0; k < N; k = k + 1) begin
        expI[k]     = prevI[k];
        expQ[k]     = prevQ[k];
        expI[k + N] = currI[k];
        expQ[k + N] = currQ[k];
      end
    end
  endtask

  task update_prev_from_curr;
    begin
      for (k = 0; k < N; k = k + 1) begin
        prevI[k] = currI[k];
        prevQ[k] = currQ[k];
      end
    end
  endtask

  task wait_fft_ready;
    integer t;
    begin
      t = 0;
      while (!fft_in_ready) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando fft_in_ready @t=%0t", $time);
          $finish;
        end
      end
    end
  endtask

  // Drive con burbujas, pero SIN burbuja en el ultimo sample
  // y asegurando fft_in_ready antes del ultimo sample (para no perder frame)
  task drive_os_block_bubbly;
    integer idx;
    integer t;
    integer bubble;
    begin
      idx = 0;
      t   = 0;

      i_valid = 1'b0;
      i_xI    = 0;
      i_xQ    = 0;

      while (idx < N) begin
        @(negedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout enviando bloque (idx=%0d) @t=%0t", idx, $time);
          $finish;
        end

        if (os_in_ready) begin
          rng = lfsr_next(rng);
          bubble = (rng[7:0] < BUBBLE_THR);

          if (idx == N-1) begin
            wait_fft_ready();
            bubble = 0;
          end

          if (!bubble) begin
            i_valid = 1'b1;
            i_xI    = currI[idx];
            i_xQ    = currQ[idx];
            idx     = idx + 1;
          end else begin
            i_valid = 1'b0;
            i_xI    = 0;
            i_xQ    = 0;
          end
        end else begin
          i_valid = 1'b0;
          i_xI    = 0;
          i_xQ    = 0;
        end
      end

      @(negedge i_clk);
      i_valid = 1'b0;
      i_xI    = 0;
      i_xQ    = 0;
    end
  endtask

  // Captura y checkea:
  // - os_fft_valid length == 32 (relacionado a os_fft_start)
  // - espera ifft_start
  // - captura 32 ifft_valid y hace VM contra exp[]
  task capture_vm_and_len_checks;
    input integer frame_id;

    integer t;
    integer cnt_os_valid;
    integer cnt_ifft_valid;

    integer idx;
    integer di, dq;

    reg seen_os_start;
    reg seen_ifft_start;
    reg too_long;

    begin
      // -------------------------
      // 1) esperar os_fft_start
      // -------------------------
      t = 0;
      seen_os_start = 0;
      while (!seen_os_start) begin
        @(posedge i_clk);
        if (os_fft_start) seen_os_start = 1;
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando os_fft_start frame %0d @t=%0t", frame_id, $time);
          $finish;
        end
      end

      // -------------------------
      // 2) contar os_fft_valid (debe ser 32)
      // -------------------------
      t = 0;
      while (!os_fft_valid) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando os_fft_valid frame %0d @t=%0t", frame_id, $time);
          $finish;
        end
      end

      // ya estamos en valid=1; contar desde este ciclo
      cnt_os_valid = 0;
      too_long     = 0;
      while (os_fft_valid && !too_long) begin
        cnt_os_valid = cnt_os_valid + 1;
        if (cnt_os_valid > (NFFT + 4)) begin
          too_long = 1;
        end
        @(posedge i_clk);
      end

      if (too_long) begin
        err_cnt = err_cnt + 1;
        $display("ERROR: os_fft_valid demasiado largo frame %0d (cnt=%0d) @t=%0t",
                 frame_id, cnt_os_valid, $time);
      end else if (cnt_os_valid != NFFT) begin
        err_cnt = err_cnt + 1;
        $display("ERROR: frame %0d os_fft_valid len=%0d (exp=%0d) @t=%0t",
                 frame_id, cnt_os_valid, NFFT, $time);
      end

      // -------------------------
      // 3) esperar ifft_start
      // -------------------------
      t = 0;
      seen_ifft_start = 0;
      while (!seen_ifft_start) begin
        @(posedge i_clk);
        if (ifft_start) seen_ifft_start = 1;
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando ifft_start frame %0d @t=%0t", frame_id, $time);
          $finish;
        end
      end

      // -------------------------
      // 4) capturar 32 ifft_valid y VM
      // -------------------------
      idx = 0;
      cnt_ifft_valid = 0;
      t = 0;

      while (idx < NFFT) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout capturando ifft_valid frame %0d idx=%0d @t=%0t",
                   frame_id, idx, $time);
          $finish;
        end

        if (ifft_valid) begin
          cnt_ifft_valid = cnt_ifft_valid + 1;

          di = $signed(ifft_yI) - $signed(expI[idx]);
          dq = $signed(ifft_yQ) - $signed(expQ[idx]);

          if ((iabs(di) > TOL_LSB) || (iabs(dq) > TOL_LSB)) begin
            err_cnt = err_cnt + 1;
            $display("ERROR: mismatch frame %0d idx %0d @t=%0t", frame_id, idx, $time);
            $display("  got: I=%0d Q=%0d", $signed(ifft_yI), $signed(ifft_yQ));
            $display("  exp: I=%0d Q=%0d", $signed(expI[idx]), $signed(expQ[idx]));
            $display("  diff: dI=%0d dQ=%0d (tol=%0d)", di, dq, TOL_LSB);
          end

          idx = idx + 1;
        end
      end

      if (cnt_ifft_valid != NFFT) begin
        err_cnt = err_cnt + 1;
        $display("ERROR: frame %0d ifft_valid capturado=%0d (exp=%0d) @t=%0t",
                 frame_id, cnt_ifft_valid, NFFT, $time);
      end

      $display("OK: frame %0d hardcore OK (os_valid=%0d, ifft_valid=%0d) @t=%0t",
               frame_id, cnt_os_valid, cnt_ifft_valid, $time);
    end
  endtask

  // ============================================================
  // MAIN
  // ============================================================
  integer f;

  initial begin
    err_cnt = 0;
    rng     = 32'h1ACE_B00C;

    i_rst   = 1'b1;
    i_valid = 1'b0;
    i_xI    = 0;
    i_xQ    = 0;

    init_prev_zero();

    repeat (5) @(posedge i_clk);
    i_rst = 1'b0;

    for (f = 0; f < NFRAMES; f = f + 1) begin
      make_curr_block_random(f);
      build_expected_frame();

      drive_os_block_bubbly();
      capture_vm_and_len_checks(f);

      update_prev_from_curr();
    end

    if (err_cnt == 0)
      $display("HARDCORE TEST OK: %0d frames sin errores", NFRAMES);
    else
      $display("HARDCORE TEST FAIL: errores=%0d en %0d frames", err_cnt, NFRAMES);

    $finish;
  end

endmodule
