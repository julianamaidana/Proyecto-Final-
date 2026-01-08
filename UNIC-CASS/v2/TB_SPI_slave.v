`timescale 1ns/1ps

module tb_spi_slave;

  localparam integer T_HALF = 10;

  reg  rst_n;
  reg  ss_n;
  reg  sclk;
  reg  mosi;
  wire miso;

  wire [6:0]  addr_out;
  wire [7:0]  data_out;
  wire        write_enable;
  reg  [7:0]  data_in;
  wire        done;
  wire [15:0] rx_frame;

  // DUT
  spi_slave_mode0 dut (
    .rst_n(rst_n),
    .ss_n(ss_n),
    .sclk(sclk),
    .mosi(mosi),
    .miso(miso),

    .addr_out(addr_out),
    .data_out(data_out),
    .write_enable(write_enable),
    .data_in(data_in),
    .done(done),
    .rx_frame(rx_frame)
  );

  // “Sistema externo”: entrega data_in según addr_out
  always @(*) begin
    if (addr_out == 7'h05)
      data_in = 8'h55;
    else
      data_in = 8'h00;
  end

  // SPI Mode 0 tasks
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

  task spi_clock_bit;
    input  mosi_bit;
    output miso_bit;
    begin
      mosi = mosi_bit;
      #(T_HALF);
      sclk = 1'b1;   // rising: sample MOSI
      #1;
      miso_bit = miso;
      #(T_HALF-1);
      sclk = 1'b0;   // falling: update MISO
      #1;
    end
  endtask

  task send_frame;
    input  [15:0] w_mosi;
    output [15:0] w_miso;
    integer k;
    reg mb;
    begin
      w_miso = 16'h0000;
      for (k=15; k>=0; k=k-1) begin
        spi_clock_bit(w_mosi[k], mb);
        w_miso[k] = mb;
      end
    end
  endtask

  // Detectores robustos de pulsos
  reg saw_write_pulse;
  reg saw_done_pulse;

  always @(posedge write_enable or negedge rst_n) begin
    if (!rst_n) saw_write_pulse <= 1'b0;
    else        saw_write_pulse <= 1'b1;
  end

  always @(posedge done or negedge rst_n) begin
    if (!rst_n) saw_done_pulse <= 1'b0;
    else        saw_done_pulse <= 1'b1;
  end

  // Test principal
  reg [15:0] miso_word;

  initial begin
    rst_n = 0;
    ss_n  = 1;
    sclk  = 0;
    mosi  = 0;

    saw_write_pulse = 0;
    saw_done_pulse  = 0;

    #50;
    rst_n = 1;
    #20;

    // TEST 1: WRITE 0xA3 en 0x05 => 0x05A3
    saw_write_pulse = 1'b0;
    saw_done_pulse  = 1'b0;

    $display("Test 1: Escribiendo 0xA3 en Direccion 0x05 (frame=0x05A3)");
    spi_begin();
    send_frame(16'h05A3, miso_word);
    spi_end();

    #5;
    if (addr_out !== 7'h05 || data_out !== 8'hA3) begin
      $display("  -> FALLO: addr_out=%02h data_out=%02h (esperado 05 / A3)", addr_out, data_out);
      $stop;
    end
    if (!saw_write_pulse) begin
      $display("  -> FALLO: no se detecto pulso write_enable");
      $stop;
    end
    if (!saw_done_pulse) begin
      $display("  -> FALLO: no se detecto pulso done");
      $stop;
    end
    if (rx_frame !== 16'h05A3) begin
      $display("  -> FALLO: rx_frame=%04h esperado 05A3", rx_frame);
      $stop;
    end

    $display("  -> EXITO: WRITE OK. addr_out=0x%02h data_out=0x%02h rx_frame=%04h", addr_out, data_out, rx_frame);

    #50;

    // TEST 2: READ de 0x05 => 0x8500, esperamos 0x55 en MISO
    saw_done_pulse = 1'b0;

    $display("Test 2: Leyendo Direccion 0x05 (frame=0x8500) esperando 0x55 en MISO");
    spi_begin();
    send_frame(16'h8500, miso_word);
    spi_end();

    #5;
    if (!saw_done_pulse) begin
      $display("  -> FALLO: no se detecto pulso done en READ");
      $stop;
    end
    if (miso_word[7:0] !== 8'h55) begin
      $display("  -> FALLO: READ miso[7:0]=%02h esperado 55 (full=%04h)", miso_word[7:0], miso_word);
      $stop;
    end

    $display("  -> EXITO: READ OK. miso[7:0]=0x%02h (full=%04h)", miso_word[7:0], miso_word);
    $display("ALL TESTS PASSED.");
    $finish;
  end

endmodule
