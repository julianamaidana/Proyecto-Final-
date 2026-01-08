module os_circular_buffer #(
    parameter WN = 16  // Ancho de datos
)(
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [WN-1:0] data_in,

    output reg  valid_out_fft,
    output reg  [WN-1:0] data_out_fft,
    output reg  start_fft // Pulso de inicio
);

    // Memoria circular de 32 posiciones
    reg [WN-1:0] mem [0:31]; 
    
    // Puntero de escritura (apunta a donde se escribe el dato entrante)
    // 5 bits para contar hasta 31
    reg [4:0] wr_ptr; 
    
    // se utiliza para resetar cada 16 datos
    reg [3:0] cnt_in; 

    // Puntero de LECTURA para enviar a la FFT
    reg [5:0] rd_count; // Cuenta de 0 a 31 para la salida
    reg [4:0] rd_addr;  // Dirección real calculada

    // Estados
    localparam S_FILL = 0;
    localparam S_SEND = 1;
    reg state;

    
    
    reg [4:0] start_read_addr;

    always @(posedge clk) begin
        if (rst) begin
            wr_ptr <= 0;
            cnt_in <= 0;
            state <= S_FILL;
            valid_out_fft <= 0;
        end else begin
            
            case (state)
                S_FILL: begin
                    valid_out_fft <= 0;
                    start_fft <= 0;

                    if (valid_in) begin
                        mem[wr_ptr] <= data_in;
                        wr_ptr <= wr_ptr + 1; // Se da la vuelta solo (31->0)
                        cnt_in <= cnt_in + 1;

                        // Si junté 16 muestras
                        if (cnt_in == 15) begin
                            state <= S_SEND;
                            rd_count <= 0;
                            start_fft <= 1; // Aviso a la FFT
                            
                            // CALCULAR DONDE EMPIEZA A LEER
                            // Si terminé de escribir en 15 -> wr_ptr pasará a 16.
                            // Quiero leer: [16..31] (Viejos) y luego [0..15] (Nuevos).
                            // O sea, empiezo a leer desde mi wr_ptr ACTUAL (que es 16).
                            // Si terminé de escribir en 31 -> wr_ptr pasará a 0.
                            // Quiero leer: [0..15] (Viejos) y luego [16..31] (Nuevos).
                            // O sea, empiezo a leer desde mi wr_ptr ACTUAL (que es 0).
                            
                            // CONCLUSIÓN MÁGICA:
                            // La dirección de inicio SIEMPRE es el valor actual de wr_ptr + 1.
                            // (Porque wr_ptr ya avanzó).
                            start_read_addr <= wr_ptr + 1; 
                        end
                    end
                end

                S_SEND: begin
                    start_fft <= 0;
                    valid_out_fft <= 1;
                    
                    // Lógica Circular de Lectura
                    // Quiero leer 32 datos secuenciales empezando desde start_read_addr
                    // rd_addr = (base + offset) % 32
                    rd_addr = start_read_addr + rd_count[4:0]; 
                    
                    data_out_fft <= mem[rd_addr];
                    
                    rd_count <= rd_count + 1;

                    if (rd_count == 31) begin
                        state <= S_FILL;
                        cnt_in <= 0; // Reseteo contador de bloque
                        // ¡NO RESETEO wr_ptr! Sigue girando.
                        // ¡NO COPIO NADA! Los datos quedan ahí.
                    end
                end
            endcase
        end
    end
endmodule