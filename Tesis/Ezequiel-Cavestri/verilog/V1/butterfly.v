module butterfly #(
  parameter integer W        = 16, // ancho fijo
  parameter integer FRAC     = 14, // bits frac
  parameter integer SCALE_EN = 1   
)(
  input  wire signed [W-1:0] i_a_re, 
  input  wire signed [W-1:0] i_a_im, 
  input  wire signed [W-1:0] i_b_re, 
  input  wire signed [W-1:0] i_b_im, 
  input  wire signed [W-1:0] i_w_re, // W real (twiddle)
  input  wire signed [W-1:0] i_w_im, // W imag (twiddle)
  output wire signed [W-1:0] o_y0_re, 
  output wire signed [W-1:0] o_y0_im, 
  output wire signed [W-1:0] o_y1_re, 
  output wire signed [W-1:0] o_y1_im  
);

  // t = B * W
  wire signed [W-1:0] t_re;
  wire signed [W-1:0] t_im;

  complex_mult #(
    .W    (W),
    .FRAC (FRAC)
  ) u_cmul (
    .i_ar (i_b_re),
    .i_ai (i_b_im),
    .i_br (i_w_re),
    .i_bi (i_w_im),
    .o_pr (t_re),
    .o_pi (t_im)
  );

  // Y0 = A + t ; Y1 = A - t  
  wire signed [W:0] y0_re_w;
  wire signed [W:0] y0_im_w;
  wire signed [W:0] y1_re_w;
  wire signed [W:0] y1_im_w;

  assign y0_re_w = $signed(i_a_re) + $signed(t_re);
  assign y0_im_w = $signed(i_a_im) + $signed(t_im);
  assign y1_re_w = $signed(i_a_re) - $signed(t_re);
  assign y1_im_w = $signed(i_a_im) - $signed(t_im);

  generate
    if (SCALE_EN) begin : g_scale
      // escala 1/2
      assign o_y0_re = y0_re_w[W:1];
      assign o_y0_im = y0_im_w[W:1];
      assign o_y1_re = y1_re_w[W:1];
      assign o_y1_im = y1_im_w[W:1];
    end else begin : g_noscale
      // sin escala
      assign o_y0_re = y0_re_w[W-1:0];
      assign o_y0_im = y0_im_w[W-1:0];
      assign o_y1_re = y1_re_w[W-1:0];
      assign o_y1_im = y1_im_w[W-1:0];
    end
  endgenerate

endmodule
