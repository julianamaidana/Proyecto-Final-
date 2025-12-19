`timescale 1ns/1ps

module tb_channel;

    // ============================================================
    // 1. PARAMETROS
    // ============================================================
    parameter DWIDTH = 9;
    parameter SNR_WIDTH = 11;
    
    reg clk;
    reg rst;
    reg signed [DWIDTH-1:0] In_I;
    reg signed [DWIDTH-1:0] In_Q;
    reg signed [SNR_WIDTH-1:0] sigma_scale;

    // Estas señales vienen del módulo (Salidas), así que son WIRE
    wire signed [DWIDTH-1:0] Out_I;
    wire signed [DWIDTH-1:0] Out_Q;

    // Variables reales para matemática (Verilog las soporta en simulación)
    real fase;
    real amplitud; 
    
    // ============================================================
    // 3. GENERACIÓN DE RELOJ (100 MHz)
    // ============================================================
    initial clk = 0;
    always #5 clk = ~clk; // Periodo 10ns

    // ============================================================
    // 4. INSTANCIA DEL MÓDULO (DUT)
    // ============================================================
    channel_with_noise #(
        .DWIDTH(DWIDTH),
        .SNR_WIDTH(SNR_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst(rst),
        .In_I(In_I),
        .In_Q(In_Q),
        .sigma_scale(sigma_scale),
        .Out_I(Out_I),
        .Out_Q(Out_Q)
    );

    // ============================================================
    // 5. ESTÍMULOS
    // ============================================================
    initial begin
        // Inicializar variables
        fase = 0.0;
        amplitud = 10000.0;

        // --- A. Inicialización ---
        rst = 1;
        In_I = 0;
        In_Q = 0;
        sigma_scale = 0;
        
        #100;
        rst = 0; // Soltamos reset
        #100;

        // --- B. PRUEBA 1: Ruido Puro (Entrada Cero) ---
        $display("Iniciando Prueba 1: Ruido Puro...");
        In_I = 0;
        In_Q = 0;
        // Sigma aprox 0.5 (512 en Q10)
        sigma_scale = 11'd512; 
        
        #2000;

        // --- C. PRUEBA 2: Señal Limpia (Sin Ruido) ---
        $display("Iniciando Prueba 2: Senal Sin Ruido...");
        sigma_scale = 0; // Apagamos el ruido
        
        #5000;

        // --- D. PRUEBA 3: Señal + Ruido ---
        $display("Iniciando Prueba 3: Senal CON Ruido...");
        sigma_scale = 11'd1000;  // Ruido fuerte        
        #5000;
        
        // --- E. PRUEBA 4: Saturación ---
        $display("Iniciando Prueba 4: Test de Saturacion...");
        sigma_scale = 0;
        amplitud = 30000.0; // Casi al limite
        
        #2000;
        
        $display("Fin de la simulacion.");
        $stop;
    end

    // ============================================================
    // 6. GENERADOR DE ONDA SINUSOIDAL
    // ============================================================
    always @(posedge clk) begin
        if (!rst) begin
            // Conversión de real a entero para la FPGA
            In_I <= $rtoi(amplitud * $sin(fase));
            In_Q <= $rtoi(amplitud * $cos(fase));
            
            // Avanzar fase
            fase = fase + 0.1; 
        end
    end

endmodule