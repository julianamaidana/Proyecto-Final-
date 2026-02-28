`timescale 1ns/1ps

module tb_validation_system_backpressure;

  // Parámetros
  localparam integer DWIDTH    = 9;
  localparam integer NFFT      = 32;
  localparam integer PART_N    = 16;
  localparam signed [DWIDTH-1:0] ONE_Q7 = 9'sd128;

  reg clk, rst;
  reg signed [10:0] sigma_scale;
  reg bypass_tx;
  reg test_data_valid;
  reg signed [DWIDTH-1:0] test_data_I, test_data_Q;
  reg signed [DWIDTH-1:0] i_W0_re, i_W0_im, i_W1_re, i_W1_im;
  
  wire ifft_valid_out;
  wire signed [DWIDTH-1:0] ifft_out_I, ifft_out_Q;
  wire fifo_full_dbg; // Señal de estado (Ya no es alarma)

  top_validation dut (
    .clk(clk), .rst(rst), .sigma_scale(sigma_scale),
    .bypass_tx(bypass_tx), .test_data_I(test_data_I), .test_data_Q(test_data_Q),
    .test_data_valid(test_data_valid),
    .i_W0_re(i_W0_re), .i_W0_im(i_W0_im), .i_W1_re(i_W1_re), .i_W1_im(i_W1_im),
    .ifft_valid_out(ifft_valid_out), .ifft_out_I(ifft_out_I), .ifft_out_Q(ifft_out_Q),
    .o_fifo_full_dbg(fifo_full_dbg), 
    .fft_valid_out(), .fft_out_I(), .fft_out_Q(), .y_valid(), .y_I(), .y_Q()
  );

  // Clock 100MHz
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  integer sample_cnt = 0;
  integer valid_frames = 0;

  initial begin
    // 1. Inicialización
    rst = 1;
    bypass_tx = 0;       // SISTEMA COMPLETO REAL
    sigma_scale = 0;
    
    // Identidad
    i_W0_re = ONE_Q7; i_W0_im = 0;
    i_W1_re = 0;      i_W1_im = 0;
    
    #100 rst = 0;
    $display("=== INICIO SIMULACION: Backpressure Activo ===");
    $display("Objetivo: Verificar que el sistema PAUSA sin perder datos.");
    
    // Corremos suficiente tiempo para ver múltiples pausas y arranques
    // Si la FIFO es de 1024 y el déficit es ~200, se llenará cada ~5 frames.
    #200000; 
    
    $display("=== FIN SIMULACION ===");
    $display("Total Frames Procesados Exitosamente: %0d", valid_frames);
    $stop;
  end

  // Monitor de Salida IFFT
  always @(posedge clk) begin
    if (ifft_valid_out) begin
        // Solo mostramos un mensaje por frame para no saturar la consola
        if (sample_cnt == 0) begin
             // $display("[IFFT OUT] Frame %0d procesado @ %0t", valid_frames, $time);
        end
        
        sample_cnt = sample_cnt + 1;
        if (sample_cnt == NFFT) begin
            sample_cnt = 0;
            valid_frames = valid_frames + 1;
        end
    end
  end

  // MONITOR DE ESTADO DE FIFO (Backpressure)
  // Detecta flancos de subida (PAUSA) y bajada (RESUME)
  reg prev_full;
  always @(posedge clk) begin
    if (rst) prev_full <= 0;
    else begin
        if (fifo_full_dbg && !prev_full) begin
            $display("[CONTROL DE FLUJO] t=%0t | FIFO LLENA -> Pausando Transmisor (TX WAIT)", $time);
        end else if (!fifo_full_dbg && prev_full) begin
            $display("[CONTROL DE FLUJO] t=%0t | FIFO LIBERADA -> Reanudando Transmisor (TX RUN)", $time);
        end
        prev_full <= fifo_full_dbg;
    end
  end

endmodule