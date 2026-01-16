module qpsk_slicer #(
  parameter int W    = 16,   // ancho de y_re/y_im
  parameter int FRAC = 15,   // bits fraccionales del formato fijo
  // amplitud del símbolo decidido en tu Q-format: +1.0 = 1<<FRAC
  parameter logic signed [W-1:0] AMP = (1'sb1 <<< FRAC),
  // umbral (normalmente 0)
  parameter logic signed [W-1:0] TH  = '0
)(
  input  logic                 clk,
  input  logic                 rst,

  input  logic                 i_valid,
  input  logic                 i_first,
  input  logic                 i_last,
  input  logic signed [W-1:0]  i_y_re,
  input  logic signed [W-1:0]  i_y_im,

  output logic                 o_valid,
  output logic                 o_first,
  output logic                 o_last,

  // bits decididos (1 si negativo, 0 si positivo)
  output logic                 o_bI_hat,
  output logic                 o_bQ_hat,

  // símbolo decidido reconstruido (±AMP)
  output logic signed [W-1:0]  o_yhat_re,
  output logic signed [W-1:0]  o_yhat_im
);

  logic negI, negQ;

  always_ff @(posedge clk) begin
    if (rst) begin
      o_valid   <= 1'b0;
      o_first   <= 1'b0;
      o_last    <= 1'b0;
      o_bI_hat  <= 1'b0;
      o_bQ_hat  <= 1'b0;
      o_yhat_re <= '0;
      o_yhat_im <= '0;
    end else begin
      // por defecto
      o_valid <= 1'b0;
      o_first <= 1'b0;
      o_last  <= 1'b0;

      if (i_valid) begin
        // decisión por signo respecto a TH
        negI = (i_y_re < TH);
        negQ = (i_y_im < TH);

        o_bI_hat <= negI;
        o_bQ_hat <= negQ;

        // reconstrucción del símbolo: si es negativo -> -AMP, si no -> +AMP
        o_yhat_re <= negI ? -AMP : AMP;
        o_yhat_im <= negQ ? -AMP : AMP;

        o_valid <= 1'b1;
        o_first <= i_first;
        o_last  <= i_last;
      end
    end
  end

endmodule
