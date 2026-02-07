`timescale 1ns / 1ps

module projection_unit #(
    parameter integer W    = 16,
    parameter integer NFFT = 32,    // Tamaño total de la FFT (2N)
    parameter integer KEEP = 16     // Cuántas muestras guardamos (N)
)(
    input  wire clk,
    input  wire rst,

    // Entrada desde la IFFT del Gradiente
    input  wire i_valid,
    input  wire i_last,
    input  wire signed [W-1:0] i_data_re,
    input  wire signed [W-1:0] i_data_im,

    // Salida hacia la FFT (para actualizar pesos)
    output reg  o_valid,
    output reg  o_last,
    output reg  signed [W-1:0] o_data_re,
    output reg  signed [W-1:0] o_data_im
);

    // Función para calcular bits del contador
    function integer clog2;
        input integer value;
        begin
            value = value - 1;
            for (clog2 = 0; value > 0; clog2 = clog2 + 1)
                value = value >> 1;
        end
    endfunction

    localparam integer CNT_BITS = clog2(NFFT);
    reg [CNT_BITS:0] counter;

    always @(posedge clk) begin
        if (rst) begin
            o_valid   <= 0;
            o_last    <= 0;
            o_data_re <= 0;
            o_data_im <= 0;
            counter   <= 0;
        end else begin
            // Por defecto
            o_valid <= 0;
            o_last  <= 0;

            if (i_valid) begin
                o_valid <= 1;
                
                // --- LÓGICA DE PROYECCIÓN ---
                // Si estamos en la primera mitad (0 a KEEP-1): PASA EL DATO
                if (counter < KEEP) begin
                    o_data_re <= i_data_re;
                    o_data_im <= i_data_im;
                end 
                // Si estamos en la segunda mitad (KEEP a NFFT-1): MUTE (CERO)
                else begin
                    o_data_re <= 0;
                    o_data_im <= 0;
                end

                // Manejo de Last y Counter
                if (i_last || counter == NFFT-1) begin
                    o_last  <= 1;
                    counter <= 0;
                end else begin
                    counter <= counter + 1;
                end
            end
        end
    end

endmodule