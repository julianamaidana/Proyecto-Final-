`timescale 1ns/1ps

module tb_spi_slave;

  reg  rst_n;
  reg  ss_n;
  reg  sclk;
  reg  mosi;
  wire miso;

  reg [15:0] mosi_w, miso_w;
  reg [15:0] mosi_r, miso_r;

  integer k;

  top_spi_regs dut (
    .rst_n(rst_n),
    .ss_n(ss_n),
    .sclk(sclk),
    .mosi(mosi),
    .miso(miso)
  );

  localparam integer T_HALF = 10;

  task spi_clock_bit;
    input  mosi_bit;
    output miso_bit;
    reg    miso_bit;
    begin
      mosi = mosi_bit;

      #(T_HALF);
      sclk = 1'b1;
      #1;
      miso_bit = miso;

      #(T_HALF);
      sclk = 1'b0;
      #1;
    end
  endtask

  task spi_transfer16;
    input  [15:0] w_mosi;
    output [15:0] w_miso;
    reg mb;
    begin
      w_miso = 16'h0000;
      for (k = 15; k >= 0; k = k - 1) begin
        spi_clock_bit(w_mosi[k], mb);
        w_miso[k] = mb;
      end
    end
  endtask

  task spi_begin;
    begin
      sclk = 1'b0;
      #5;
      ss_n = 1'b0;
      #5;
    end
  endtask

  task spi_end;
    begin
      sclk = 1'b0;
      #5;
      ss_n = 1'b1;
      #20;
    end
  endtask

  initial begin
    rst_n = 0;
    ss_n  = 1;
    sclk  = 0;
    mosi  = 0;

    #50;
    rst_n = 1;
    #20;

    // WRITE: RW=1, ADDR=0x12, DATA=0x3A -> 16'h923A
    mosi_w = {1'b1, 7'h12, 8'h3A};

    spi_begin();
    spi_transfer16(mosi_w, miso_w);
    spi_end();

    #50;

    if (dut.regfile[7'h12] !== 8'h3A) begin
      $display("FAIL WRITE: regfile[0x12]=%02h expected 3A", dut.regfile[7'h12]);
      $stop;
    end else begin
      $display("PASS WRITE: regfile[0x12]=%02h", dut.regfile[7'h12]);
    end

    // READ: RW=0, ADDR=0x12, dummy=0x00 -> 16'h1200
    mosi_r = {1'b0, 7'h12, 8'h00};

    spi_begin();
    spi_transfer16(mosi_r, miso_r);
    spi_end();

    if (miso_r[7:0] !== 8'h3A) begin
      $display("FAIL READ: miso_r[7:0]=%02h expected 3A (full=%04h, hi=%02h lo=%02h)",
               miso_r[7:0], miso_r, miso_r[15:8], miso_r[7:0]);
      $stop;
    end else begin
      $display("PASS READ: miso_r[7:0]=%02h (full=%04h)", miso_r[7:0], miso_r);
    end

    $display("ALL TESTS PASSED.");
    $finish;
  end

endmodule
// SPI/tb_slave.v