// top_spi.v
module top_spi_regs (
  input  wire rst_n,
  input  wire ss_n,
  input  wire sclk,
  input  wire mosi,
  output wire miso
);

  wire [6:0]  addr_out;
  wire [7:0]  data_out;
  wire        write_enable;
  wire [7:0]  data_in;
  wire        done;
  wire [15:0] rx_frame;

  // 128 registros de 8 bits
  reg [7:0] regfile [0:127];

  // lectura combinacional
  assign data_in = regfile[addr_out];

  // escritura en negedge para evitar carrera (write_enable/addr/data salen en posedge)
  always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
      // opcional: init si querés evitar X al inicio (para sim)
      // integer i;
      // for (i=0; i<128; i=i+1) regfile[i] <= 8'h00;
    end else if (!ss_n && write_enable) begin
      regfile[addr_out] <= data_out;
    end
  end

  spi_slave_param_mode0 #(
    .FRAME_BITS(16),
    .ADDR_BITS(7),
    .DATA_BITS(8),
    .RW_BIT(15),
    .ADDR_MSB(14),
    .ADDR_LSB(8),
    .DATA_MSB(7),
    .DATA_LSB(0),
    .RESP_START_BIT(7),
    .MSB_FIRST(1)
  ) u_spi (
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

endmodule
