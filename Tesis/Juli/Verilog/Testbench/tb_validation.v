`timescale 1ns / 1ps

module tb_validation;

    // ============================================================
    // 1. SEÑALES DEL TESTBENCH
    // ============================================================
    reg clk = 0;
    reg rst = 1;
    
    // Control de Ruido: Lo ponemos en 0 para validar lógica
    reg signed [10:0] sigma_scale = 11'sd0; 

    // Control del MUX de prueba
    reg bypass_tx = 0;           // 0: Usa Tx Real, 1: Usa test_data
    reg signed [8:0] test_data_I = 0;
    reg signed [8:0] test_data_Q = 0;

    // Salidas a observar
    wire fft_valid_out;
    wire signed [8:0] fft_out_I;
    wire signed [8:0] fft_out_Q;

    // Generación de Reloj (100 MHz -> 10ns periodo)
    always #5 clk = ~clk;

    // ============================================================
    // 2. INSTANCIA DEL TOP DE VALIDACIÓN
    // ============================================================
    top_validation #(
        .DWIDTH(9),
        .SNR_WIDTH(11),
        .N_PART(16),
        .NFFT(32)
    ) u_dut (
        .clk(clk),
        .rst(rst),
        .sigma_scale(sigma_scale),
        .bypass_tx(bypass_tx),
        .test_data_I(test_data_I),
        .test_data_Q(test_data_Q),
        .fft_valid_out(fft_valid_out),
        .fft_out_I(fft_out_I),
        .fft_out_Q(fft_out_Q)
    );

    // ============================================================
    // 3. PROCESO DE PRUEBA
    // ============================================================
    initial begin
        // --- A. INICIALIZACIÓN ---
        $display("=== INICIO SIMULACION ===");
        clk = 0;
        rst = 1;
        bypass_tx = 0; // Arrancamos desconectados del manual
        sigma_scale = 0; // SIN RUIDO

        #100;       // Esperamos 100ns
        rst = 0;    // Soltamos Reset
        $display("Reset liberado.");

        // --- B. PRUEBA DE BYPASS (DC - CONSTANTE) ---
        // Objetivo: Verificar que la FFT pone todo en el Bin 0
        $display("--- TEST 1: INYECCION DE CONSTANTE (DC) ---");
        
        bypass_tx = 1;         // Activamos modo manual
        test_data_I = 9'sd100; // Valor constante 100
        test_data_Q = 9'sd0;

        // Esperamos a que salgan un par de paquetes válidos de FFT
        // (El buffer tarda en llenarse, la FFT tarda en calcular)
        repeat (5) @(posedge fft_valid_out); 
        
        $display("Ya deberían haber salido datos de la FFT (Mirar Waveform).");
        
        // --- C. PRUEBA DE SISTEMA COMPLETO (TX QPSK) ---
        // Objetivo: Verificar que el Tx y el Canal (Ideal) pasan datos al buffer
        $display("--- TEST 2: SISTEMA COMPLETO (TX QPSK) ---");
        
        #200; // Pequeña pausa
        bypass_tx = 0; // Desactivamos manual -> Entra el QPSK
        
        // Dejamos correr un buen tiempo para ver la modulación
        #10000; 
        
        $display("=== FIN SIMULACION ===");
        $stop;
    end

endmodule