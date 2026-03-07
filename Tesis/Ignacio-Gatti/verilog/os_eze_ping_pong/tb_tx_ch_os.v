`timescale 1ns/1ps
`default_nettype none

module tb_top_tx_ch_os;

  localparam integer N  = 16;
  localparam integer WN = 9;

  reg  clk_fast;
  reg  rst;
  reg  enable_div;
  reg  [10:0] sigma_scale;

  wire clk_low;

  wire signed [WN-1:0] tx_I_dbg, tx_Q_dbg;
  wire signed [WN-1:0] ch_I_dbg, ch_Q_dbg;

  wire os_overflow;
  wire os_start;
  wire os_valid;
  wire signed [WN-1:0] os_I;
  wire signed [WN-1:0] os_Q;

  // clk_fast = 100 MHz
  initial clk_fast = 1'b0;
  always #5 clk_fast = ~clk_fast;

  top_tx_ch_os #(
    .N(N),
    .WN(WN)
  ) dut (
    .clk_fast   (clk_fast),
    .rst        (rst),
    .enable_div (enable_div),
    .sigma_scale(sigma_scale),

    .clk_low    (clk_low),
    .tx_I_dbg   (tx_I_dbg),
    .tx_Q_dbg   (tx_Q_dbg),
    .ch_I_dbg   (ch_I_dbg),
    .ch_Q_dbg   (ch_Q_dbg),

    .os_overflow(os_overflow),
    .os_start   (os_start),
    .os_valid   (os_valid),
    .os_I       (os_I),
    .os_Q       (os_Q)
  );

  initial begin
    $dumpfile("tb_top_tx_ch_os.vcd");
    $dumpvars(0, tb_top_tx_ch_os);
  end

  initial begin
    enable_div  = 1'b1;      // clave: clk_low debe togglear durante reset

    // "sin ruido" usando top_ch:
    //sigma_scale = 11'd0;
    sigma_scale = 11'd64;

    rst = 1'b1;
    wait (clk_low === 1'b0 || clk_low === 1'b1);
    repeat (4) @(posedge clk_low);
    rst = 1'b0;

    repeat (2500) @(posedge clk_fast);

    $display("[%0t] END: overflow=%0d", $time, os_overflow);
    $finish;
  end

  // Monitor: os_start cada 2N ciclos de clk_fast (steady-state)
  integer cyc_since_start;
  reg started;

  initial begin
    cyc_since_start = 0;
    started = 1'b0;
  end

  always @(posedge clk_fast) begin
    if (rst) begin
      started <= 1'b0;
      cyc_since_start <= 0;
    end else begin
      if (os_overflow)
        $display("[%0t] **OVERFLOW**", $time);

      if (os_start) begin
        if (started && (cyc_since_start != (2*N-1)))
          $display("[%0t] WARN: periodo start=%0d (esperado %0d)", $time, cyc_since_start, 2*N-1);

        started <= 1'b1;
        cyc_since_start <= 0;

        $display("[%0t] FRAME_START  os_I=%0d os_Q=%0d", $time, os_I, os_Q);
      end else if (started) begin
        cyc_since_start <= cyc_since_start + 1;
      end
    end
  end

endmodule

`default_nettype wire
