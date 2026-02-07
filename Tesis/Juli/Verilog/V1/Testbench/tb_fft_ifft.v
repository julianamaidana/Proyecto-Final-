`timescale 1ns/1ps

module tb_fft_ifft_identity;

  localparam integer NFFT = 32;
  localparam integer LOGN = 5;

  // Vamos a testear en Q10 para minimizar error por cuantización
  localparam integer NB  = 17;
  localparam integer NBF = 10;

  localparam integer TIMEOUT_CYC = 50000;

  // clk/rst
  reg i_clk;
  reg i_rst;

  // FFT ports
  reg                     fft_valid;
  reg signed [NB-1:0]     fft_xI, fft_xQ;
  wire                    fft_ready;
  wire                    fft_start;
  wire                    fft_o_valid;
  wire signed [NB-1:0]    fft_yI, fft_yQ;

  // IFFT ports
  reg                     ifft_valid;
  reg signed [NB-1:0]     ifft_xI, ifft_xQ;
  wire                    ifft_ready;
  wire                    ifft_start;
  wire                    ifft_o_valid;
  wire signed [NB-1:0]    ifft_yI, ifft_yQ;

  // =========================
  // DUTs
  // =========================
  fft_ifft #(
    .NFFT(NFFT), .LOGN(LOGN),
    .NB_IN(NB),  .NBF_IN(NBF),
    .NB_W(17),   .NBF_W(10),
    .NB_OUT(NB), .NBF_OUT(NBF),
    .SCALE_STAGE(0)
  ) u_fft (
    .i_clk(i_clk),
    .i_rst(i_rst),
    .i_valid(fft_valid),
    .i_xI(fft_xI),
    .i_xQ(fft_xQ),
    .i_inverse(1'b0),
    .o_in_ready(fft_ready),
    .o_start(fft_start),
    .o_valid(fft_o_valid),
    .o_yI(fft_yI),
    .o_yQ(fft_yQ)
  );

  fft_ifft #(
    .NFFT(NFFT), .LOGN(LOGN),
    .NB_IN(NB),  .NBF_IN(NBF),
    .NB_W(17),   .NBF_W(10),
    .NB_OUT(NB), .NBF_OUT(NBF),
    .SCALE_STAGE(0)
  ) u_ifft (
    .i_clk(i_clk),
    .i_rst(i_rst),
    .i_valid(ifft_valid),
    .i_xI(ifft_xI),
    .i_xQ(ifft_xQ),
    .i_inverse(1'b1),   // IFFT
    .o_in_ready(ifft_ready),
    .o_start(ifft_start),
    .o_valid(ifft_o_valid),
    .o_yI(ifft_yI),
    .o_yQ(ifft_yQ)
  );

  // =========================
  // Clock
  // =========================
  initial i_clk = 1'b0;
  always #5 i_clk = ~i_clk;

  // =========================
  // Memories
  // =========================
  reg signed [NB-1:0] xI [0:NFFT-1];
  reg signed [NB-1:0] xQ [0:NFFT-1];

  reg signed [NB-1:0] XI [0:NFFT-1];
  reg signed [NB-1:0] XQ [0:NFFT-1];

  reg signed [NB-1:0] yI [0:NFFT-1];
  reg signed [NB-1:0] yQ [0:NFFT-1];

  integer i;
  integer err_cnt;

  // abs diff <= 1 LSB
  function integer abs_i;
    input integer v;
    begin
      abs_i = (v < 0) ? -v : v;
    end
  endfunction

  // =========================
  // Helpers: wait with timeout
  // =========================
  task wait_ready_fft;
    integer t;
    begin
      t = 0;
      while (!fft_ready) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando fft_ready @t=%0t", $time);
          $finish;
        end
      end
    end
  endtask

  task wait_ready_ifft;
    integer t;
    begin
      t = 0;
      while (!ifft_ready) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando ifft_ready @t=%0t", $time);
          $finish;
        end
      end
    end
  endtask

  task wait_start_fft;
    integer t;
    begin
      t = 0;
      while (!fft_start) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando fft_start @t=%0t", $time);
          $finish;
        end
      end
    end
  endtask

  task wait_start_ifft;
    integer t;
    begin
      t = 0;
      while (!ifft_start) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando ifft_start @t=%0t", $time);
          $finish;
        end
      end
    end
  endtask

  // =========================
  // Build random small Q10 input (safe)
  // Range aprox: [-256, 255] => valor [-0.25, 0.249]
  // =========================
  task build_input_random;
    integer seed;
    integer v;
    begin
      seed = 32'h1234_5678;
      for (i = 0; i < NFFT; i = i + 1) begin
        // LCG simple
        seed = (1103515245 * seed + 12345);
        v = (seed >>> 16) & 9'h1FF;   // 0..511
        v = v - 256;                  // -256..255
        xI[i] = v[NB-1:0];

        seed = (1103515245 * seed + 12345);
        v = (seed >>> 16) & 9'h1FF;
        v = v - 256;
        xQ[i] = v[NB-1:0];
      end
    end
  endtask

  // =========================
  // Drive FFT input frame
  // =========================
  task drive_fft_frame;
    integer n;
    begin
      wait_ready_fft();
      n = 0;

      fft_valid = 1'b0;
      fft_xI    = 0;
      fft_xQ    = 0;

      while (n < NFFT) begin
        @(negedge i_clk);
        if (fft_ready) begin
          fft_valid = 1'b1;
          fft_xI    = xI[n];
          fft_xQ    = xQ[n];
          n = n + 1;
        end else begin
          fft_valid = 1'b0;
          fft_xI    = 0;
          fft_xQ    = 0;
        end
      end

      @(negedge i_clk);
      fft_valid = 1'b0;
      fft_xI    = 0;
      fft_xQ    = 0;
    end
  endtask

  // Capture FFT output to XI/XQ
  task capture_fft_frame;
    integer idx;
    integer t;
    begin
      wait_start_fft();
      idx = 0; t = 0;

      while (idx < NFFT) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout capturando FFT @t=%0t", $time);
          $finish;
        end
        if (fft_o_valid) begin
          XI[idx] = fft_yI;
          XQ[idx] = fft_yQ;
          idx = idx + 1;
        end
      end
    end
  endtask

  // =========================
  // Drive IFFT frame with XI/XQ
  // =========================
  task drive_ifft_frame;
    integer n;
    begin
      wait_ready_ifft();
      n = 0;

      ifft_valid = 1'b0;
      ifft_xI    = 0;
      ifft_xQ    = 0;

      while (n < NFFT) begin
        @(negedge i_clk);
        if (ifft_ready) begin
          ifft_valid = 1'b1;
          ifft_xI    = XI[n];
          ifft_xQ    = XQ[n];
          n = n + 1;
        end else begin
          ifft_valid = 1'b0;
          ifft_xI    = 0;
          ifft_xQ    = 0;
        end
      end

      @(negedge i_clk);
      ifft_valid = 1'b0;
      ifft_xI    = 0;
      ifft_xQ    = 0;
    end
  endtask

  // Capture IFFT output to yI/yQ
  task capture_ifft_frame;
    integer idx;
    integer t;
    begin
      wait_start_ifft();
      idx = 0; t = 0;

      while (idx < NFFT) begin
        @(posedge i_clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout capturando IFFT @t=%0t", $time);
          $finish;
        end
        if (ifft_o_valid) begin
          yI[idx] = ifft_yI;
          yQ[idx] = ifft_yQ;
          idx = idx + 1;
        end
      end
    end
  endtask

  // =========================
  // Compare y vs x (tol 1 LSB)
  // =========================
  task compare_identity;
    integer di, dq;
    begin
      for (i = 0; i < NFFT; i = i + 1) begin
        di = $signed(yI[i]) - $signed(xI[i]);
        dq = $signed(yQ[i]) - $signed(xQ[i]);

        if (abs_i(di) > 1 || abs_i(dq) > 1) begin
          err_cnt = err_cnt + 1;
          $display("ERROR idx=%0d", i);
          $display("  x:  I=%0d Q=%0d", $signed(xI[i]), $signed(xQ[i]));
          $display("  y:  I=%0d Q=%0d", $signed(yI[i]), $signed(yQ[i]));
          $display("  d:  I=%0d Q=%0d", di, dq);
        end
      end
    end
  endtask

  // =========================
  // MAIN
  // =========================
  initial begin
    err_cnt = 0;

    i_rst      = 1'b1;
    fft_valid  = 1'b0;
    ifft_valid = 1'b0;
    fft_xI     = 0; fft_xQ = 0;
    ifft_xI    = 0; ifft_xQ = 0;

    repeat (5) @(posedge i_clk);
    i_rst = 1'b0;

    $display("TEST IDENTITY: x -> FFT -> IFFT -> x (tol=1 LSB en Q10)");
    build_input_random();

    drive_fft_frame();
    capture_fft_frame();

    drive_ifft_frame();
    capture_ifft_frame();

    compare_identity();

    if (err_cnt == 0) $display("TEST OK: identidad validada");
    else              $display("TEST FAIL: errores=%0d", err_cnt);

    repeat (5) @(posedge i_clk);
    $finish;
  end

endmodule
