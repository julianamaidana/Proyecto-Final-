`timescale 1ns/1ps

module qpsk_slicer #(
  parameter integer W    = 16,
  parameter integer FRAC = 14,               // <-- ojo: si querés 16384, FRAC debe ser 14 si es 15 queda en 32768
  parameter signed [W-1:0] TH = 0
)(
  input  wire                i_clk,
  input  wire                i_rst,

  input  wire                i_valid,
  input  wire                i_first,
  input  wire                i_last,
  input  wire signed [W-1:0] i_y_re,
  input  wire signed [W-1:0] i_y_im,

  output reg                 o_valid,
  output reg                 o_first,
  output reg                 o_last,
  output reg                 o_bI_hat,
  output reg                 o_bQ_hat,
  output reg signed [W-1:0]  o_yhat_re,
  output reg signed [W-1:0]  o_yhat_im
);

  // +1.0 en Q(FRAC)
  localparam signed [W-1:0] AMP = (1 <<< FRAC);

  always @(posedge i_clk) begin
    if (i_rst) begin
      o_valid   <= 1'b0;
      o_first   <= 1'b0;
      o_last    <= 1'b0;
      o_bI_hat  <= 1'b0;
      o_bQ_hat  <= 1'b0;
      o_yhat_re <= {W{1'b0}};
      o_yhat_im <= {W{1'b0}};
    end else begin
      if (i_valid) begin
        // señales de control
        o_valid <= 1'b1;
        o_first <= i_first;
        o_last  <= i_last;

        // bits: 1 si negativo
        o_bI_hat <= (i_y_re < TH);
        o_bQ_hat <= (i_y_im < TH);

        // símbolo decidido: ±AMP
        o_yhat_re <= ( (i_y_re < TH) ? -AMP : AMP );
        o_yhat_im <= ( (i_y_im < TH) ? -AMP : AMP );

      end else begin
        // limpieza cuando no hay dato válido
        o_valid   <= 1'b0;
        o_first   <= 1'b0;
        o_last    <= 1'b0;
        o_bI_hat  <= 1'b0;
        o_bQ_hat  <= 1'b0;
        o_yhat_re <= {W{1'b0}};
        o_yhat_im <= {W{1'b0}};
      end
    end
  end

endmodule
