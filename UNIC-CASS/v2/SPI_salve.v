`timescale 1ns/1ps

module spi_slave_mode0 #(
  parameter integer FRAME_BITS = 16,
  parameter integer ADDR_BITS  = 7,
  parameter integer DATA_BITS  = 8,

  // Trama final (cuando ya recibiste los 16 bits)
  // [15]=RW, [14:8]=ADDR, [7:0]=DATA
  parameter integer RW_BIT    = 15, // 1=READ, 0=WRITE
  parameter integer ADDR_MSB  = 14,
  parameter integer ADDR_LSB  = 8,
  parameter integer DATA_MSB  = 7,
  parameter integer DATA_LSB  = 0,

  // Fin de cabecera (1+ADDR_BITS = 8 bits) => bit_cnt==7
  parameter integer HDR_LAST_BIT = 7
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

  localparam integer CNT_W    = $clog2(FRAME_BITS);
  localparam integer TXC_W    = $clog2(DATA_BITS);
  localparam integer LAST_BIT = FRAME_BITS-1;

  reg [FRAME_BITS-1:0] rx_shift;
  reg [FRAME_BITS-1:0] tx_shift;

  reg [CNT_W-1:0] bit_cnt;

  // Latches
  reg rw_latched;
  reg [ADDR_BITS-1:0] addr_latched;

  // Handshake RX->TX (toggle) sin doble driver
  reg rd_toggle;

  // TX control (solo en el always de TX)
  reg [TXC_W-1:0] tx_cnt;
  reg             tx_active;
  reg             rd_toggle_q;

  // MISO tri-state cuando no seleccionado
  assign miso = (!ss_n) ? tx_shift[FRAME_BITS-1] : 1'bz;

  // =========================
  // RX: posedge sclk (CPHA=0)
  // =========================
  reg [FRAME_BITS-1:0] next_rx;

  always @(posedge sclk or negedge rst_n) begin
    if (!rst_n) begin
      rx_shift     <= {FRAME_BITS{1'b0}};
      bit_cnt      <= {CNT_W{1'b0}};
      rw_latched   <= 1'b0;
      addr_latched <= {ADDR_BITS{1'b0}};
      rd_toggle    <= 1'b0;

      addr_out     <= {ADDR_BITS{1'b0}};
      data_out     <= {DATA_BITS{1'b0}};
      write_enable <= 1'b0;
      done         <= 1'b0;
      rx_frame     <= {FRAME_BITS{1'b0}};
    end else begin
      // pulsos por defecto
      write_enable <= 1'b0;
      done         <= 1'b0;

      if (ss_n) begin
        rx_shift <= {FRAME_BITS{1'b0}};
        bit_cnt  <= {CNT_W{1'b0}};
      end else begin
        // Shift-left: entra MOSI por LSB (MSB-first)
        next_rx  = {rx_shift[FRAME_BITS-2:0], mosi};
        rx_shift <= next_rx;

        // --- FIN DE CABECERA (8 bits recibidos) ---
        // IMPORTANTE: a este punto, la cabecera está en next_rx[7:0]
        if (bit_cnt == HDR_LAST_BIT[CNT_W-1:0]) begin
          rw_latched   <= next_rx[HDR_LAST_BIT];               // bit7
          addr_latched <= next_rx[HDR_LAST_BIT-1:0];           // bits[6:0]
          addr_out     <= next_rx[HDR_LAST_BIT-1:0];

          // Si es READ (RW=1) pedimos al TX que cargue data_in
          if (next_rx[HDR_LAST_BIT] == 1'b1)
            rd_toggle <= ~rd_toggle;
        end

        // --- FIN DE FRAME (16 bits) ---
        if (bit_cnt == LAST_BIT[CNT_W-1:0]) begin
          rx_frame <= next_rx;

          // Decodificación final (ya alineada en [15:0])
          addr_out <= next_rx[ADDR_MSB:ADDR_LSB];
          data_out <= next_rx[DATA_MSB:DATA_LSB];

          // RW=0 => WRITE
          if (next_rx[RW_BIT] == 1'b0)
            write_enable <= 1'b1;

          done    <= 1'b1;
          bit_cnt <= {CNT_W{1'b0}};
        end else begin
          bit_cnt <= bit_cnt + 1'b1;
        end
      end
    end
  end

  // =========================
  // TX: negedge sclk (CPHA=0)
  // =========================
  always @(negedge sclk or negedge rst_n) begin
    if (!rst_n) begin
      tx_shift    <= {FRAME_BITS{1'b0}};
      tx_cnt      <= {TXC_W{1'b0}};
      tx_active   <= 1'b0;
      rd_toggle_q <= 1'b0;
    end else begin
      if (ss_n) begin
        tx_shift    <= {FRAME_BITS{1'b0}};
        tx_cnt      <= {TXC_W{1'b0}};
        tx_active   <= 1'b0;
        rd_toggle_q <= rd_toggle;
      end else begin
        // Si RX pidió READ (toggle cambió), cargamos data_in
        if (rd_toggle_q != rd_toggle) begin
          rd_toggle_q <= rd_toggle;

          // data_in en bits altos para salir MSB-first por MISO
          tx_shift  <= {data_in, {(FRAME_BITS-DATA_BITS){1'b0}}}; // ej: 0x5500
          tx_cnt    <= {TXC_W{1'b0}};
          tx_active <= 1'b1;
        end else if (tx_active) begin
          // Desplazamos 1 bit por negedge para preparar el próximo bit
          if (tx_cnt < (DATA_BITS-1)) begin
            tx_shift <= {tx_shift[FRAME_BITS-2:0], 1'b0};
            tx_cnt   <= tx_cnt + 1'b1;
          end else begin
            tx_active <= 1'b0; // ya se emitieron 8 bits
          end
        end
      end
    end
  end

endmodule
