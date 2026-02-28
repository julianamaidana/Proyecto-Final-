`timescale 1ns/1ps

module tb_os_buffer_ping_pong_ce;

  parameter integer N  = 16;
  parameter integer WN = 9;

  reg                   i_clk;
  reg                   i_rst;
  reg                   i_valid;
  reg                   i_ce_in;
  reg  signed [WN-1:0]  i_xI;
  reg  signed [WN-1:0]  i_xQ;

  wire                  o_overflow;
  wire                  o_fft_start;
  wire                  o_fft_valid;
  wire signed [WN-1:0]  o_fft_xI;
  wire signed [WN-1:0]  o_fft_xQ;

  // DUT
  os_buffer #(.N(N), .WN(WN)) dut (
    .i_clk(i_clk),
    .i_rst(i_rst),
    .i_valid(i_valid),
    .i_ce_in(i_ce_in),
    .i_xI(i_xI),
    .i_xQ(i_xQ),
    .o_overflow(o_overflow),
    .o_fft_start(o_fft_start),
    .o_fft_valid(o_fft_valid),
    .o_fft_xI(o_fft_xI),
    .o_fft_xQ(o_fft_xQ)
  );

  // Clock 100MHz
  initial i_clk = 1'b0;
  always #5 i_clk = ~i_clk;

  // CE: pulso 1 ciclo cada 2 clocks (mundo lento = clk/2)
  reg ce_tog;
  always @(posedge i_clk) begin
    if (i_rst) begin
      ce_tog <= 1'b0;
      i_ce_in <= 1'b0;
    end else begin
      ce_tog  <= ~ce_tog;
      i_ce_in <= ~ce_tog;  // 1,0,1,0,...
    end
  end

  // -------------------------
  // Helpers
  // -------------------------
  integer k;
  integer err_cnt;

  function signed [WN-1:0] clip_s;
    input integer x;
    integer maxv, minv, y;
    begin
      maxv = (1 << (WN-1)) - 1;
      minv = -(1 << (WN-1));
      y = x;
      if (y > maxv) y = maxv;
      if (y < minv) y = minv;
      clip_s = y[WN-1:0];
    end
  endfunction

  // Scoreboard: frame = [prev | curr]
  reg signed [WN-1:0] prevI [0:N-1];
  reg signed [WN-1:0] prevQ [0:N-1];
  reg signed [WN-1:0] currI [0:N-1];
  reg signed [WN-1:0] currQ [0:N-1];
  reg signed [WN-1:0] expI  [0:(2*N)-1];
  reg signed [WN-1:0] expQ  [0:(2*N)-1];

  task init_prev_zero;
    begin
      for (k = 0; k < N; k = k + 1) begin
        prevI[k] = 0;
        prevQ[k] = 0;
      end
    end
  endtask

  task make_curr_block;
    input integer seed;
    begin
      for (k = 0; k < N; k = k + 1) begin
        currI[k] = clip_s(seed + 3*k + 1);
        currQ[k] = clip_s(seed - 2*k - 1);
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

  // Drive N samples, pero SOLO en ciclos donde i_ce_in=1 (1 cada 2 clocks)
  task drive_N_samples_ce;
    integer idx;
    begin
      idx = 0;
      while (idx < N) begin
        @(negedge i_clk);
        if (i_ce_in) begin
          i_valid = 1'b1;
          i_xI    = currI[idx];
          i_xQ    = currQ[idx];
          idx     = idx + 1;
        end else begin
          // mantener estable / no enviar muestra efectiva
          i_valid = 1'b1; // puede quedar en 1, igual el DUT solo toma cuando ce=1
          i_xI    = i_xI;
          i_xQ    = i_xQ;
        end
      end

      @(negedge i_clk);
      i_valid = 1'b0;
      i_xI    = 0;
      i_xQ    = 0;
    end
  endtask

  // Checker: sincroniza con o_fft_start y lee con #1 para evitar NBA race
  task check_one_frame;
    input integer frame_id;
    integer idx;
    begin
      wait (o_fft_start === 1'b1);

      @(posedge i_clk);
      #1;
      if (o_fft_valid !== 1'b1) begin
        err_cnt = err_cnt + 1;
        $display("[ERR] frame %0d: luego de o_fft_start se esperaba o_fft_valid=1 @t=%0t",
                 frame_id, $time);
        disable check_one_frame;
      end

      for (idx = 0; idx < (2*N); idx = idx + 1) begin
        if (o_fft_valid !== 1'b1) begin
          err_cnt = err_cnt + 1;
          $display("[ERR] frame %0d: o_fft_valid cayo antes de 2N (idx=%0d) @t=%0t",
                   frame_id, idx, $time);
          disable check_one_frame;
        end

        #1;
        if ((o_fft_xI !== expI[idx]) || (o_fft_xQ !== expQ[idx])) begin
          err_cnt = err_cnt + 1;
          $display("[ERR] frame %0d idx %0d mismatch @t=%0t", frame_id, idx, $time);
          $display("      got: I=%0d Q=%0d", $signed(o_fft_xI), $signed(o_fft_xQ));
          $display("      exp: I=%0d Q=%0d", $signed(expI[idx]), $signed(expQ[idx]));
        end

        if (idx != (2*N-1)) begin
          @(posedge i_clk);
          #1;
        end
      end

      @(posedge i_clk);
      #1;
      if (o_fft_valid !== 1'b0) begin
        err_cnt = err_cnt + 1;
        $display("[ERR] frame %0d: o_fft_valid duro mas de 2N @t=%0t", frame_id, $time);
      end else begin
        $display("[OK] frame %0d correcto (2N=%0d muestras) @t=%0t", frame_id, 2*N, $time);
      end
    end
  endtask

  // -------------------------
  // TEST A: funcional
  // -------------------------
  task test_A_functional;
    integer f;
    begin
      $display("=== TEST A: funcional (entrada con CE=clk/2) ===");
      init_prev_zero();

      for (f = 0; f < 5; f = f + 1) begin
        make_curr_block(10*f + 7);
        build_expected_frame();

        drive_N_samples_ce();
        check_one_frame(f);

        update_prev_from_curr();
        repeat (6) @(posedge i_clk);
      end

      if (o_overflow !== 1'b0) begin
        err_cnt = err_cnt + 1;
        $display("[ERR] TEST A: overflow se prendio, no deberia @t=%0t", $time);
      end
    end
  endtask

  // -------------------------
  // TEST B: streaming continuo con CE
  // -------------------------
  integer s;
  reg overflow_seen;

  always @(posedge i_clk) begin
    if (i_rst) overflow_seen <= 1'b0;
    else if (o_overflow && !overflow_seen) begin
      overflow_seen <= 1'b1;
      $display("[WARN] o_overflow se activo (se dropeo al menos 1 sample) @t=%0t", $time);
    end
  end

  task test_B_streaming;
    integer cycles_fast;
    integer in_count;
    begin
      $display("=== TEST B: streaming continuo (i_valid=1, muestras solo con CE) ===");

      i_valid = 1'b1;
      in_count = 0;

      // correr 2000 clocks "rápidos"
      cycles_fast = 2000;
      for (s = 0; s < cycles_fast; s = s + 1) begin
        @(negedge i_clk);
        if (i_ce_in) begin
          i_xI = clip_s(1000 + in_count);
          i_xQ = clip_s(-(1000 + in_count));
          in_count = in_count + 1;
        end
      end

      @(negedge i_clk);
      i_valid = 1'b0;
      i_xI    = 0;
      i_xQ    = 0;

      repeat (50) @(posedge i_clk);

      if (!overflow_seen) $display("[OK] TEST B: NO overflow en streaming con CE (corrio %0d ciclos fast)", cycles_fast);
      else                $display("[WARN] TEST B: hubo overflow aun con CE");
    end
  endtask

  // -------------------------
  // MAIN
  // -------------------------
  initial begin
    err_cnt = 0;
    i_rst   = 1'b1;
    i_valid = 1'b0;
    i_xI    = 0;
    i_xQ    = 0;

    repeat (10) @(posedge i_clk);
    i_rst = 1'b0;
    repeat (5) @(posedge i_clk);

    test_A_functional();
    test_B_streaming();

    if (err_cnt == 0) $display("=== TB RESULT: PASS (sin errores) ===");
    else              $display("=== TB RESULT: FAIL (errores=%0d) ===", err_cnt);

    $finish;
  end

endmodule
