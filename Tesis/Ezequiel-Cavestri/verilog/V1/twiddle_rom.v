module twiddle_rom #(
  parameter integer NFFT = 32, // puntos FFT
  parameter integer W    = 16, // ancho
  parameter integer FRAC = 14  // bits frac 
)(
  input  wire [$clog2(NFFT/2)-1:0] i_addr, // addr 0..NFFT/2-1
  output reg  signed [W-1:0]       o_re,   // twiddle real
  output reg  signed [W-1:0]       o_im    // twiddle imag
);

  localparam integer DEPTH = NFFT/2;

  reg signed [W-1:0] rom_re [0:DEPTH-1]; // tabla cos()
  reg signed [W-1:0] rom_im [0:DEPTH-1]; // tabla -sin()

  // carga desde .mem (hex)
  initial begin
    $readmemh("tw_re.mem", rom_re);
    $readmemh("tw_im.mem", rom_im);
  end

  // lectura combinacional
  always @(*) begin
    o_re = rom_re[i_addr];
    o_im = rom_im[i_addr];
  end

endmodule
