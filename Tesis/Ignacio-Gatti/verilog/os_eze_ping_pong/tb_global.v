`timescale 1ns/1ps
`default_nettype none

module tb_top_tx_ch_os_fifo;

  localparam integer N        = 16;
  localparam integer WN       = 9;
  localparam integer FIFO_AW  = 8;

  reg  clk_fast;
  reg  rst;
  reg  enable_div;
  reg  [10:0] sigma_scale;

  wire clk_low;

  wire signed [WN-1:0] tx_I_dbg, tx_Q_dbg;
  wire signed [WN-1:0] ch_I_dbg, ch_Q_dbg;

  wire os_overflow, os_start, os_valid;
  wire signed [WN-1:0] os_I, os_Q;

  reg  fifo_rd_en;
  wire fifo_full, fifo_empty, fifo_valid, fifo_overflow;
  wire [2*WN:0] fifo_dout;
  wire [FIFO_AW:0] fifo_count;

  // decode {start, I, Q}
  wire start_d;
  wire signed [WN-1:0] I_d;
  wire signed [WN-1:0] Q_d;

  assign start_d = fifo_dout[2*WN];
  assign I_d     = fifo_dout[2*WN-1:WN];
  assign Q_d     = fifo_dout[WN-1:0];

  integer samp_since_start;
  reg started;
  integer max_count;

  // clk_fast 100MHz
  initial clk_fast = 1'b0;
  always #5 clk_fast = ~clk_fast;

  top_tx_ch_os_fifo #(
    .N(N), .WN(WN), .FIFO_AW(FIFO_AW)
  ) dut (
    .clk_fast     (clk_fast),
    .rst          (rst),
    .enable_div   (enable_div),
    .sigma_scale  (sigma_scale),

    .fifo_rd_en   (fifo_rd_en),

    .clk_low      (clk_low),
    .tx_I_dbg     (tx_I_dbg),
    .tx_Q_dbg     (tx_Q_dbg),
    .ch_I_dbg     (ch_I_dbg),
    .ch_Q_dbg     (ch_Q_dbg),

    .os_overflow  (os_overflow),
    .os_start     (os_start),
    .os_valid     (os_valid),
    .os_I         (os_I),
    .os_Q         (os_Q),

    .fifo_full    (fifo_full),
    .fifo_empty   (fifo_empty),
    .fifo_valid   (fifo_valid),
    .fifo_overflow(fifo_overflow),
    .fifo_dout    (fifo_dout),
    .fifo_count   (fifo_count)
  );

  initial begin
    $dumpfile("tb_top_tx_ch_os_fifo.vcd");
    $dumpvars(0, tb_top_tx_ch_os_fifo);
  end

  initial begin
    enable_div  = 1'b1;

    // Ruido:
    //  sigma_scale = 0   => sin ruido
    //  sigma_scale > 0   => con ruido
    sigma_scale = 11'd0;
    //sigma_scale = 11'd64;

    fifo_rd_en = 1'b0;

    rst = 1'b1;
    wait (clk_low === 1'b0 || clk_low === 1'b1);
    repeat (4) @(posedge clk_low);
    rst = 1'b0;

    // “FFT ideal”: leo siempre (SIN STALL)
    fifo_rd_en = 1'b1;

    repeat (2500) @(posedge clk_fast);

    $display("[%0t] END: os_overflow=%0d fifo_overflow=%0d fifo_max_count=%0d",
             $time, os_overflow, fifo_overflow, max_count);
    $finish;
  end

  initial begin
    samp_since_start = 0;
    started = 1'b0;
    max_count = 0;
  end

  always @(posedge clk_fast) begin
    if (rst) begin
      samp_since_start <= 0;
      started <= 1'b0;
      max_count <= 0;
    end else begin
      if (fifo_count > max_count) max_count <= fifo_count;

      if (os_overflow)   $display("[%0t] **OS_OVERFLOW**", $time);
      if (fifo_overflow) $display("[%0t] **FIFO_OVERFLOW**", $time);

      // ojo: fifo_valid viene 1 ciclo después del read aceptado
      if (fifo_valid) begin
        if (start_d) begin
          // Con este conteo, lo esperado es 2N-1 (=31 si N=16)
          if (started && (samp_since_start != (2*N-1)))
            $display("[%0t] WARN: periodo FRAME_START=%0d (esperado %0d)",
                     $time, samp_since_start, 2*N-1);

          started <= 1'b1;
          samp_since_start <= 0;

          $display("[%0t] FRAME_START  I=%0d Q=%0d  fifo_count=%0d",
                   $time, I_d, Q_d, fifo_count);
        end else if (started) begin
          samp_since_start <= samp_since_start + 1;
        end
      end
    end
  end

endmodule

`default_nettype wire
