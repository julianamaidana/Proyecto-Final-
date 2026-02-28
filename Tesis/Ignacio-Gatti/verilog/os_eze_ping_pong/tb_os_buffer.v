`timescale 1ns/1ps
`default_nettype none

module tb_os_buffer_v2;

  localparam integer N  = 16;
  localparam integer WN = 9;

  reg  clk_fast;
  wire clk_low;
  reg  rst;
  reg  enable;

  // inputs (low)
  reg                  i_valid;
  reg  signed [WN-1:0] i_i;
  reg  signed [WN-1:0] i_q;

  // outputs (fast)
  wire                 o_overflow;
  wire                 o_start;
  wire                 o_valid;
  wire signed [WN-1:0] o_i;
  wire signed [WN-1:0] o_q;

  // 100 MHz
  initial clk_fast = 1'b0;
  always #5 clk_fast = ~clk_fast;

  // divider del proyecto (sin rst, con enable)
  clock_div2 u_div2 (
    .i_clk_fast(clk_fast),
    .i_enable  (enable),
    .o_clk_low (clk_low)
  );

  // DUT
  os_buffer #(
    .N (N),
    .WN(WN)
  ) dut (
    .i_clk_low (clk_low),
    .i_clk_fast(clk_fast),
    .i_rst     (rst),

    .i_valid   (i_valid),
    .i_i       (i_i),
    .i_q       (i_q),

    .o_overflow(o_overflow),
    .o_start   (o_start),
    .o_valid   (o_valid),
    .o_i       (o_i),
    .o_q       (o_q)
  );

  // dump
  initial begin
    $dumpfile("tb_os_buffer_v2.vcd");
    $dumpvars(0, tb_os_buffer_v2);
  end

  // reset/enable
  initial begin
    enable  = 1'b1;     // clave: clk_low necesita togglear durante reset
    rst     = 1'b1;

    i_valid = 1'b0;
    i_i     = 0;
    i_q     = 0;

    wait (clk_low === 1'b0 || clk_low === 1'b1);
    repeat (4) @(posedge clk_low);
    rst = 1'b0;

    repeat (800) @(posedge clk_fast);

    $display("[%0t] TEST END: overflow=%0d", $time, o_overflow);
    $finish;
  end

  // stimulus: 1..16 cyclic, Q=-I (drive en negedge)
  reg signed [WN-1:0] k;

  always @(negedge clk_low) begin
    if (rst) begin
      i_valid <= 1'b0;
      i_i     <= 0;
      i_q     <= 0;
      k       <= 1;
    end else begin
      i_valid <= 1'b1;
      i_i     <= k;
      i_q     <= -k;

      if (k == 16) k <= 1;
      else         k <= k + 1;
    end
  end

  // monitor: start alineado con valid
  integer out_cnt;
  integer idx_print;
  reg overflow_d;

  always @(posedge clk_fast) begin
    if (rst) begin
      out_cnt    <= 0;
      overflow_d <= 1'b0;
    end else begin
      overflow_d <= o_overflow;
      if (!overflow_d && o_overflow)
        $display("[%0t] OVERFLOW RISE", $time);

      if (o_valid) begin
        idx_print = (o_start) ? 0 : out_cnt;

        if (o_start)
          $display("[%0t] FRAME_START", $time);

        $display("[%0t] OUT[%0d] I=%0d Q=%0d", $time, idx_print, o_i, o_q);

        // check rápido: OUT[0..15]=0, OUT[16]=1 en primer frame
        if (idx_print == 16 && o_i !== 1)
          $display("[%0t] WARN: expected OUT[16]=1, got %0d", $time, o_i);

        if (idx_print == (2*N-1))
          $display("[%0t] FRAME_END", $time);

        out_cnt <= idx_print + 1;
      end
    end
  end

endmodule

`default_nettype wire
