module fifo #(
    parameter DATA_WIDTH = 18,      
    parameter ADDR_WIDTH = 8         // 2^8 = 256 posiciones (Suficiente para FFT de 32)
)(
    input  wire                   clk,
    input  wire                   rst,      // Reset síncrono
    
    // Interfaz de Escritura (Viene de la PRBS)
    input  wire [DATA_WIDTH-1:0]  din,
    input  wire                   wr_en,
    
    // Interfaz de Lectura (Viene del Overlap)
    input  wire                   rd_en,
    output reg  [DATA_WIDTH-1:0]  dout,
    
    // Banderas de estado
    output wire                   full,
    output wire                   empty,
    output reg                    valid,
    output reg                    overflow   // Indica si perdimos datos por no frenar la PRBS
);

    localparam DEPTH = 1 << ADDR_WIDTH;

    // Memoria inferida (Vivado la mapeará a BRAM o Distributed RAM según el tamaño)
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    reg [ADDR_WIDTH-1:0] wr_ptr = 0;
    reg [ADDR_WIDTH-1:0] rd_ptr = 0;
    reg [ADDR_WIDTH:0]   count = 0;  // Un bit extra para diferenciar full de empty

    // Banderas
    assign empty = (count == 0);
    assign full  = (count == DEPTH);

    // Lógica de Escritura y Lectura
    always @(posedge clk) begin
        if (rst) begin
            wr_ptr   <= 0;
            rd_ptr   <= 0;
            count    <= 0;
            overflow <= 0;
            dout     <= 0;
        end else begin
            // Escritura: Si llega dato de la PRBS
            if (wr_en) begin
                if (!full) begin
                    mem[wr_ptr] <= din;
                    wr_ptr      <= wr_ptr + 1;
                end else begin
                    overflow <= 1'b1; // Error: Memoria llena, dato perdido
                end
            end

            // Lectura: Si el Overlap pide dato
            if (rd_en && !empty) begin
                dout   <= mem[rd_ptr];
                rd_ptr <= rd_ptr + 1;
            end

            // Gestión del contador de ocupación
            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1; // Solo escritura
                2'b01: count <= count - 1; // Solo lectura
                default: count <= count;   // Nada o ambas (se compensan)
            endcase
        end
    end

endmodule