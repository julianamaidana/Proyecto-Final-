// SPI_slave.v
// spi_slave_param_mode0_v.v  (Verilog-2001 compatible)
// SPI Slave Mode 0: capture MOSI @posedge SCLK, update MISO @negedge SCLK
module spi_slave_param_mode0 #(
  parameter integer FRAME_BITS = 16,
  parameter integer ADDR_BITS  = 7,
  parameter integer DATA_BITS  = 8,

  // bit positions in frame [FRAME_BITS-1:0]
  parameter integer RW_BIT     = 15,
  parameter integer ADDR_MSB   = 14,
  parameter integer ADDR_LSB   = 8,
  parameter integer DATA_MSB   = 7,
  parameter integer DATA_LSB   = 0,

  // load read response after header received (default: after 8 bits -> index 7)
  parameter integer RESP_START_BIT = 7,

  // 1 = MSB-first, 0 = LSB-first
  parameter MSB_FIRST = 1
)(
  input  wire                  rst_n,
  input  wire                  ss_n,
  input  wire                  sclk,
  input  wire                  mosi,
  output wire                  miso,

  output reg  [ADDR_BITS-1:0]  addr_out,
  output reg  [DATA_BITS-1:0]  data_out,
  output reg                   write_enable,
  input  wire [DATA_BITS-1:0]  data_in,

  output reg                   done,
  output reg  [FRAME_BITS-1:0] rx_frame
);

  reg [FRAME_BITS-1:0] rx_shift;
  reg [FRAME_BITS-1:0] tx_shift;

  function integer clog2;
    input integer value;
    integer i;
    begin
      clog2 = 0;
      for (i = value-1; i > 0; i = i >> 1)
        clog2 = clog2 + 1;
    end
  endfunction

  localparam integer CNT_W = (FRAME_BITS <= 2) ? 1 : clog2(FRAME_BITS);
  reg [CNT_W-1:0] bit_cnt;

  localparam [CNT_W-1:0] LAST_BIT   = FRAME_BITS-1;
  localparam [CNT_W-1:0] RESP_LATCH = RESP_START_BIT;
  localparam [CNT_W-1:0] RESP_LOAD  = RESP_START_BIT + 1;

  reg rw_latched;
  reg [ADDR_BITS-1:0] addr_latched;

  reg [FRAME_BITS-1:0] next_rx;
  reg [FRAME_BITS-1:0] frame_full;

  wire tx_bit;
  assign tx_bit = (MSB_FIRST) ? tx_shift[FRAME_BITS-1] : tx_shift[0];

  assign miso = (!ss_n) ? tx_bit : 1'bz;

  always @(posedge ss_n or negedge rst_n) begin
    if (!rst_n) begin
      rx_shift     <= {FRAME_BITS{1'b0}};
      tx_shift     <= {FRAME_BITS{1'b0}};
      bit_cnt      <= {CNT_W{1'b0}};
      rw_latched   <= 1'b0;
      addr_latched <= {ADDR_BITS{1'b0}};
      addr_out     <= {ADDR_BITS{1'b0}};
      data_out     <= {DATA_BITS{1'b0}};
      write_enable <= 1'b0;
      done         <= 1'b0;
      rx_frame     <= {FRAME_BITS{1'b0}};
    end else begin
      rx_shift     <= {FRAME_BITS{1'b0}};
      tx_shift     <= {FRAME_BITS{1'b0}};
      bit_cnt      <= {CNT_W{1'b0}};
      rw_latched   <= 1'b0;
      addr_latched <= {ADDR_BITS{1'b0}};
      write_enable <= 1'b0;
      done         <= 1'b0;
    end
  end

  // RX
  always @(posedge sclk or negedge rst_n) begin
    if (!rst_n) begin
      rx_shift     <= {FRAME_BITS{1'b0}};
      bit_cnt      <= {CNT_W{1'b0}};
      write_enable <= 1'b0;
      done         <= 1'b0;
      rx_frame     <= {FRAME_BITS{1'b0}};
      rw_latched   <= 1'b0;
      addr_latched <= {ADDR_BITS{1'b0}};
    end else if (!ss_n) begin
      write_enable <= 1'b0;
      done         <= 1'b0;

      if (MSB_FIRST) next_rx = {rx_shift[FRAME_BITS-2:0], mosi};
      else           next_rx = {mosi, rx_shift[FRAME_BITS-1:1]};

      rx_shift <= next_rx;

      if (bit_cnt == RESP_LATCH) begin
        rw_latched   <= next_rx[RW_BIT];
        addr_latched <= next_rx[ADDR_MSB:ADDR_LSB];
      end

      if (bit_cnt == LAST_BIT) begin
        frame_full = next_rx;

        rx_frame <= frame_full;
        addr_out <= frame_full[ADDR_MSB:ADDR_LSB];
        data_out <= frame_full[DATA_MSB:DATA_LSB];

        if (frame_full[RW_BIT]) write_enable <= 1'b1;
        done <= 1'b1;

        bit_cnt <= {CNT_W{1'b0}};
      end else begin
        bit_cnt <= bit_cnt + {{(CNT_W-1){1'b0}},1'b1};
      end
    end
  end

  // TX
  always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
      tx_shift <= {FRAME_BITS{1'b0}};
    end else if (!ss_n) begin
      if (bit_cnt == {CNT_W{1'b0}}) begin
        tx_shift <= {FRAME_BITS{1'b0}};
      end
      else if (bit_cnt == RESP_LOAD) begin
        if (!rw_latched) begin
          // READ: MSB-first => data_in en bits altos
          if (MSB_FIRST)
            tx_shift <= { data_in, {(FRAME_BITS-DATA_BITS){1'b0}} };
          else
            tx_shift <= { {(FRAME_BITS-DATA_BITS){1'b0}}, data_in };
        end else begin
          tx_shift <= {FRAME_BITS{1'b0}};
        end
      end else begin
        if (MSB_FIRST)
          tx_shift <= {tx_shift[FRAME_BITS-2:0], 1'b0};
        else
          tx_shift <= {1'b0, tx_shift[FRAME_BITS-1:1]};
      end
    end
  end

endmodule
