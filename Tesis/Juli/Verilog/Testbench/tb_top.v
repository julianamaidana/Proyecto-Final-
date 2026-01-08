`timescale 1ns/1ps

module tb_top;

    // ============================================================
    // 1. SEÑALES Y PARÁMETROS
    // ============================================================
    parameter DWIDTH = 9;
    parameter SNR_WIDTH = 11; 
    
    // IMPORTANTE: Definimos cuántos bits son fraccionales
    parameter DATA_F = 7; 

    reg clk;
    reg rst;
    reg signed [SNR_WIDTH-1:0] sigma_scale;

    // Salidas crudas (Enteros)
    wire signed [DWIDTH-1:0] rx_I_monitor;
    wire signed [DWIDTH-1:0] rx_Q_monitor;

    // ============================================================
    // 2. INSTANCIA DEL SISTEMA (DUT)
    // ============================================================
    top #(
        .DWIDTH(DWIDTH),
        .SNR_WIDTH(SNR_WIDTH)
    ) u_dut (
        .clk         (clk),
        .rst         (rst),
        .sigma_scale (sigma_scale),
        .rx_I        (rx_I_monitor),
        .rx_Q        (rx_Q_monitor)
    );

    // ============================================================
    // 3. CABLES ESPIA (Acceso Jerárquico)
    // ============================================================
    wire signed [DWIDTH-1:0] tx_I_clean = u_dut.tx_I_internal; 
    wire signed [DWIDTH-1:0] tx_Q_clean = u_dut.tx_Q_internal;

    // ============================================================
    // --- NUEVO: CONVERSION A REAL (FLOAT) ---
    // Esto es lo que le "dice" al simulador dónde está el punto
    // ============================================================
    
    // Variables flotantes para ver en la gráfica
    real real_tx_I, real_tx_Q;
    real real_rx_I, real_rx_Q;

    // Factor de escala: 2^7 = 128.0
    // Usamos 2.0 ** DATA_F para que sea automático
    real SCALE_FACTOR;
    
    initial SCALE_FACTOR = 2.0 ** DATA_F; // Calcula 128.0

    // Conversión constante: Divide el entero por 128.0
    always @* begin
        real_tx_I = $itor(tx_I_clean)   / SCALE_FACTOR;
        real_tx_Q = $itor(tx_Q_clean)   / SCALE_FACTOR;
        real_rx_I = $itor(rx_I_monitor) / SCALE_FACTOR;
        real_rx_Q = $itor(rx_Q_monitor) / SCALE_FACTOR;
    end

    // ============================================================
    // 4. GENERACIÓN DE RELOJ Y PROCESO
    // ============================================================
    initial clk = 0;
    always #5 clk = ~clk; 

    initial begin
        rst = 1;
        sigma_scale = 0; 
        
        #100;
        rst = 0;
        
        // FASE 1: Limpia
        #5000; 

        // FASE 2: Ruido Bajo
        sigma_scale = 11'd200; 
        #5000;

        // FASE 3: Ruido Alto
        sigma_scale = 11'd800;
        #5000;

        $stop;
    end

endmodule