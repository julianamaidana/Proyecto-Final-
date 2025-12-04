`timescale 1ns / 1ps

module prbs9 #(
    parameter [8:0] SEED = 9'h1AA // Valor inicial por defecto
)(
    input  wire clk,
    input  wire rst_n,    // Reset activo bajo (0 = reset)
    input  wire en,       // Enable: si está en 1, genera nuevo bit
    output wire bit_out   // Salida del bit generado
);

    reg [8:0] lfsr;
    wire      feedback;

    // Polinomio x^9 + x^5 + 1
    // Indices Verilog: [8] es el bit más significativo (x^9)
    // lfsr[4] corresponde a x^5
    assign feedback = lfsr[8] ^ lfsr[4];
    
    // La salida es el bit más significativo
    assign bit_out  = lfsr[8];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr <= SEED; // Cargar semilla al resetear
        end else if (en) begin
            // Desplazamiento a izquierda e inserción del feedback al final
            lfsr <= {lfsr[7:0], feedback};
        end
    end

endmodule