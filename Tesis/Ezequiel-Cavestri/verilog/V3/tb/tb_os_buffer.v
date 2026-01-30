`timescale 1ns/1ps

module tb_os_buffer;

  parameter integer N  = 16;
  parameter integer WN = 9;

  reg                   i_clk;
  reg                   i_rst;
  reg                   i_valid;
  reg  signed [WN-1:0]  i_xI;
  reg  signed [WN-1:0]  i_xQ;

  wire                  o_in_ready;
  wire                  o_fft_start;
  wire                  o_fft_valid;
  wire signed [WN-1:0]  o_fft_xI;
  wire signed [WN-1:0]  o_fft_xQ;

  os_buffer #(.N(N), .WN(WN)) dut (
    .i_clk(i_clk),
    .i_rst(i_rst),
    .i_valid(i_valid),
    .i_xI(i_xI),
    .i_xQ(i_xQ),
    .o_in_ready(o_in_ready),
    .o_fft_start(o_fft_start),
    .o_fft_valid(o_fft_valid),
    .o_fft_xI(o_fft_xI),
    .o_fft_xQ(o_fft_xQ)
  );

  // clock
  initial i_clk = 1'b0;
  always #5 i_clk = ~i_clk;

  // scoreboard
  reg signed [WN-1:0] prevI [0:N-1];
  reg signed [WN-1:0] prevQ [0:N-1];
  reg signed [WN-1:0] currI [0:N-1];
  reg signed [WN-1:0] currQ [0:N-1];
  reg signed [WN-1:0] expI  [0:2*N-1];
  reg signed [WN-1:0] expQ  [0:2*N-1];

  integer k;
  integer out_idx;
  integer frame_idx;
  reg     prev_valid;
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

  // drive en negedge (setup antes del posedge)
  task drive_one_block;
    integer in_idx;
    begin
      in_idx  = 0;
      i_valid = 0;
      i_xI    = 0;
      i_xQ    = 0;

      while (in_idx < N) begin
        @(negedge i_clk);
        if (o_in_ready) begin
          i_valid = 1;
          i_xI    = currI[in_idx];
          i_xQ    = currQ[in_idx];
          in_idx  = in_idx + 1;
        end else begin
          i_valid = 0;
          i_xI    = 0;
          i_xQ    = 0;
        end
      end

      @(negedge i_clk);
      i_valid = 0;
      i_xI    = 0;
      i_xQ    = 0;
    end
  endtask

  // -------------------------
  // MONITOR / CHECKER (FIXED)
  // -------------------------
    integer idx_now;
    reg rising;
    reg falling;

  always @(posedge i_clk) begin

    if (i_rst) begin
      out_idx    <= 0;
      frame_idx  <= 0;
      prev_valid <= 0;
      err_cnt    <= 0;
    end else begin

      rising  = (!prev_valid &&  o_fft_valid);
      falling = ( prev_valid && !o_fft_valid);

      // si arranca frame, idx_now=0; si no, usa out_idx actual
      idx_now = rising ? 0 : out_idx;

      // comparar mientras valid, usando idx_now
      if (o_fft_valid) begin
        if (idx_now >= (2*N)) begin
          err_cnt <= err_cnt + 1;
          $display("ERROR: frame %0d -> o_fft_valid excede 2N @t=%0t (idx=%0d)",
                   frame_idx, $time, idx_now);
        end else begin
          if ((o_fft_xI !== expI[idx_now]) || (o_fft_xQ !== expQ[idx_now])) begin
            err_cnt <= err_cnt + 1;
            $display("ERROR: mismatch frame %0d idx %0d @t=%0t", frame_idx, idx_now, $time);
            $display("  got: I=%0d Q=%0d", $signed(o_fft_xI), $signed(o_fft_xQ));
            $display("  exp: I=%0d Q=%0d", $signed(expI[idx_now]), $signed(expQ[idx_now]));
          end
        end

        // unica asignacion: avanzar desde idx_now
        out_idx <= idx_now + 1;
      end else begin
        // si no hay valid, mantener idx (o dejarlo como está)
        out_idx <= idx_now;
      end

      // fin de frame: cae valid => chequear longitud
      if (falling) begin
        // NOTA: out_idx en este ciclo todavia tiene el valor anterior,
        // pero idx_now ya refleja el valor correcto acumulado hasta el ultimo valid.
        // En la caida, el ultimo incremento ya ocurrió en el ciclo anterior,
        // así que out_idx debe ser 2N.
        if (out_idx != (2*N)) begin
          err_cnt <= err_cnt + 1;
          $display("ERROR: frame %0d -> longitud valid=%0d (esperado %0d) @t=%0t",
                   frame_idx, out_idx, 2*N, $time);
        end else begin
          $display("OK: Frame %0d correcto (%0d muestras)", frame_idx, out_idx);
        end

        frame_idx <= frame_idx + 1;
        update_prev_from_curr();
      end

      prev_valid <= o_fft_valid;
    end
  end

  // main
  integer f;

  initial begin
    i_rst   = 1;
    i_valid = 0;
    i_xI    = 0;
    i_xQ    = 0;

    init_prev_zero();

    repeat (5) @(posedge i_clk);
    i_rst = 0;

    for (f = 0; f < 5; f = f + 1) begin
      make_curr_block(10*f + 7);
      build_expected_frame();

      drive_one_block();

      // sync robusta por o_fft_valid
      wait (o_fft_valid == 1);
      wait (o_fft_valid == 0);
      @(posedge i_clk);
    end

    if (err_cnt == 0) $display("TEST OK: sin errores");
    else              $display("TEST FAIL: errores=%0d", err_cnt);

    $finish;
  end

endmodule
