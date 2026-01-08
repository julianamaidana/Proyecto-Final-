module complex_mult #(
    parameter integer W    = 16,
    parameter integer FRAC = 14
)(
    input  wire signed [W-1:0] i_ar,
    input  wire signed [W-1:0] i_ai,
    input  wire signed [W-1:0] i_br,
    input  wire signed [W-1:0] i_bi,
    output wire signed [W-1:0] o_pr,
    output wire signed [W-1:0] o_pi
);
    wire signed [2*W-1:0] p1 = $signed(i_ar) * $signed(i_br);
    wire signed [2*W-1:0] p2 = $signed(i_ai) * $signed(i_bi);
    wire signed [2*W-1:0] p3 = $signed(i_ar) * $signed(i_bi);
    wire signed [2*W-1:0] p4 = $signed(i_ai) * $signed(i_br);

    wire signed [2*W:0] re_w = $signed(p1) - $signed(p2);
    wire signed [2*W:0] im_w = $signed(p3) + $signed(p4);

    // scale back
    assign o_pr = re_w >>> FRAC;
    assign o_pi = im_w >>> FRAC;
endmodule