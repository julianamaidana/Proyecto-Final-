`timescale 1ns/1ps

module tb_top_validation_fft_ifft_identity;

  // ========= PARAMS =========
  localparam integer N      = 32;
  localparam integer DW     = 9;   // tu DWIDTH
  localparam integer TOL    = 1;   // tolerancia (por redondeos)

  // ========= CLK/RST =========
  reg clk = 0;
  always #50 clk = ~clk; // 10ns? ajustá a tu clock real (acá 100ns)

  reg rst_n = 0;

  // ========= DUT CONTROLS (ajustar a tu top) =========
  reg bypass_tx;

  // taps/weights (W0=1, W1=0 como decís)
  // ajustá width/formato a tu diseño
  reg signed [DW-1:0] w0_re, w0_im;
  reg signed [DW-1:0] w1_re, w1_im;

  // ========= DUT OBSERVABILITY (ajustar a tu top) =========
  // *** ESTO ES CLAVE ***
  // Señales en el boundary FFT/IFFT que querés verificar
  wire                  fft_in_valid;
  wire signed [DW-1:0]   fft_in_i;
  wire signed [DW-1:0]   fft_in_q;

  wire                  ifft_out_valid;
  wire signed [DW-1:0]  ifft_out_i;
  wire signed [DW-1:0]  ifft_out_q;

  // (Opcional) si tenés first/last mejor todavía
  wire fft_in_first, fft_in_last;
  wire ifft_out_first, ifft_out_last;

  // ========= DUT INSTANTIATION =========
  // Cambiá esto por tu instancia real y conectá puertos
  top_validation_tx_ch_os_fft_ifft_buffer_hist #(
    .DWIDTH(DW)
  ) dut (
    .clk       (clk),
    .rst_n     (rst_n),

    .bypass_tx (bypass_tx),

    .w0_re     (w0_re),
    .w0_im     (w0_im),
    .w1_re     (w1_re),
    .w1_im     (w1_im),

    // IMPORTANT: conectá los taps internos o puertos de debug
    .fft_in_valid (fft_in_valid),
    .fft_in_first (fft_in_first),
    .fft_in_last  (fft_in_last),
    .fft_in_i     (fft_in_i),
    .fft_in_q     (fft_in_q),

    .ifft_out_valid (ifft_out_valid),
    .ifft_out_first (ifft_out_first),
    .ifft_out_last  (ifft_out_last),
    .ifft_out_i     (ifft_out_i),
    .ifft_out_q     (ifft_out_q)
  );

  // ========= SCOREBOARD =========
  reg signed [DW-1:0] exp_i [0:N-1];
  reg signed [DW-1:0] exp_q [0:N-1];

  integer cap_k, chk_k;
  integer frame_in, frame_out;
  integer err_cnt;

  task scoreboard_clear;
    integer k;
    begin
      for (k=0; k<N; k=k+1) begin
        exp_i[k] = '0;
        exp_q[k] = '0;
      end
      cap_k = 0;
      chk_k = 0;
    end
  endtask

  function integer abs_int;
    input integer x;
    begin
      abs_int = (x < 0) ? -x : x;
    end
  endfunction

  // Capturo EXACTAMENTE lo que entra a FFT (lo que tu sistema “cree” que está procesando)
  always @(posedge clk) begin
    if (!rst_n) begin
      cap_k    <= 0;
      frame_in <= 0;
    end else begin
      if (fft_in_valid) begin
        exp_i[cap_k] <= fft_in_i;
        exp_q[cap_k] <= fft_in_q;

        if (cap_k == N-1) begin
          cap_k    <= 0;
          frame_in <= frame_in + 1;
        end else begin
          cap_k <= cap_k + 1;
        end
      end
    end
  end

  // Comparo lo que sale de IFFT contra lo capturado
  always @(posedge clk) begin
    if (!rst_n) begin
      chk_k     <= 0;
      frame_out <= 0;
      err_cnt   <= 0;
    end else begin
      if (ifft_out_valid) begin
        integer di, dq;
        di = ifft_out_i - exp_i[chk_k];
        dq = ifft_out_q - exp_q[chk_k];

        if (abs_int(di) > TOL || abs_int(dq) > TOL) begin
          $display("[ERR] IFFT!=IN frame_out=%0d k=%0d got=(%0d,%0d) exp=(%0d,%0d) d=(%0d,%0d) @t=%0t",
                   frame_out, chk_k,
                   ifft_out_i, ifft_out_q,
                   exp_i[chk_k], exp_q[chk_k],
                   di, dq, $time);
          err_cnt <= err_cnt + 1;
        end

        if (chk_k == N-1) begin
          chk_k     <= 0;
          frame_out <= frame_out + 1;
          $display("[INFO] frame_out=%0d done (err=%0d) @t=%0t",
                   frame_out+1, err_cnt, $time);
        end else begin
          chk_k <= chk_k + 1;
        end
      end
    end
  end

  // ========= TEST SEQUENCER =========
  initial begin
    $display("=== INICIO SIM ===");

    // reset
    bypass_tx = 1'b1;
    w0_re = 0; w0_im = 0;
    w1_re = 0; w1_im = 0;

    rst_n = 0;
    scoreboard_clear();
    repeat (20) @(posedge clk);
    rst_n = 1;
    $display("Reset liberado.");

    // ---------------------------
    // TEST 1: DC bypass (bypass_tx=1) + W0=1, W1=0
    // ---------------------------
    $display("--- TEST 1: DC bypass + W0=1 W1=0 => IFFT(FFT(x))=x ---");
    w0_re = 1; w0_im = 0;
    w1_re = 0; w1_im = 0;

    // correr algunos frames
    wait_frames(8);

    // ---------------------------
    // IMPORTANT: flush + limpiar scoreboard antes del TEST 2
    // porque cambian los datos y (probablemente) el pipeline arrastra cosas
    // ---------------------------
    scoreboard_clear();
    frame_in  = 0;
    frame_out = 0;
    err_cnt   = 0;

    // drenar pipeline un rato
    repeat (200) @(posedge clk);

    // ---------------------------
    // TEST 2: TX real (bypass_tx=0) + W0=1, W1=0
    // ---------------------------
    $display("--- TEST 2: TX real (bypass_tx=0) + W0=1 W1=0 ---");
    bypass_tx = 1'b0;

    wait_frames(10);

    $display("=== FIN SIM === frames_in=%0d frames_out=%0d err=%0d", frame_in, frame_out, err_cnt);
    $finish;
  end

  // helper: espera N frames_out completos
  task wait_frames;
    input integer nframes;
    integer target;
    begin
      target = frame_out + nframes;
      while (frame_out < target) @(posedge clk);
    end
  endtask

endmodule
