`timescale 1ns / 1ps
// =============================================================
//  axi4_affinity_top.v  –  AXI4-Full Slave Accelerator
//  DVCon India 2026 – Parallel BRAM-based Scorer
//
//  Description:
//    AXI4-Full slave wrapper around the parallel_scorer engine.
//    CPU programmes all N object embeddings (into BRAM) plus
//    metadata, then asserts START.  Hardware dispatches objects
//    in batches of N_LANES, tracks the winner, and asserts DONE.
//
//  Register Map (byte addresses relative to slave base):
//    0x0000  CTRL        [0]=START (W, self-clearing)
//    0x0004  STATUS      [0]=DONE  [1]=BUSY (R)
//    0x0008  N_OBJECTS   [15:0]           (RW)
//    0x000C  BEST_OBJECT [15:0]           (R)
//    0x0010  BEST_SCORE  [31:0]           (R)
//    0x0100 + i*4  TASK_EMB[i]  [7:0]    (RW)  i=0..EMB_DIM-1
//    0x0200 + o*4  CONF[o]      [15:0]   (RW)  o=0..MAX_OBJ-1
//    0x0300 + o*4  BOOST[o]     [15:0]   (RW)  o=0..MAX_OBJ-1
//    0x0400 + o*4  PENALTY[o]   [0]      (RW)  o=0..MAX_OBJ-1
//    0x1000 + (o*EMB_DIM + b)*4  OBJ_EMB[o][b] [7:0] (RW)
// =============================================================
module axi4_affinity_top #(
    parameter EMB_DIM     = 8,
    parameter MAX_OBJECTS = 16,
    parameter N_LANES     = 4,
    parameter AXI_ADDR_W  = 32,
    parameter AXI_DATA_W  = 32,
    parameter AXI_ID_W    = 4
)(
    input  wire                      ACLK,
    input  wire                      ARESETn,

    // ── Write Address Channel ──────────────────────────────
    input  wire [AXI_ID_W-1:0]      AWID,
    input  wire [AXI_ADDR_W-1:0]    AWADDR,
    input  wire [7:0]                AWLEN,
    input  wire [2:0]                AWSIZE,
    input  wire [1:0]                AWBURST,
    input  wire                      AWVALID,
    output reg                       AWREADY,

    // ── Write Data Channel ─────────────────────────────────
    input  wire [AXI_DATA_W-1:0]    WDATA,
    input  wire [AXI_DATA_W/8-1:0]  WSTRB,
    input  wire                      WLAST,
    input  wire                      WVALID,
    output reg                       WREADY,

    // ── Write Response Channel ─────────────────────────────
    output reg  [AXI_ID_W-1:0]      BID,
    output reg  [1:0]                BRESP,
    output reg                       BVALID,
    input  wire                      BREADY,

    // ── Read Address Channel ───────────────────────────────
    input  wire [AXI_ID_W-1:0]      ARID,
    input  wire [AXI_ADDR_W-1:0]    ARADDR,
    input  wire [7:0]                ARLEN,
    input  wire [2:0]                ARSIZE,
    input  wire [1:0]                ARBURST,
    input  wire                      ARVALID,
    output reg                       ARREADY,

    // ── Read Data Channel ──────────────────────────────────
    output reg  [AXI_ID_W-1:0]      RID,
    output reg  [AXI_DATA_W-1:0]    RDATA,
    output reg  [1:0]                RRESP,
    output reg                       RLAST,
    output reg                       RVALID,
    input  wire                      RREADY
);

    // =========================================================
    //  Derived parameters
    // =========================================================
    localparam EMB_DIM_LOG2 = $clog2(EMB_DIM);
    localparam OBJ_IDX_W    = $clog2(MAX_OBJECTS);

    // Active-high internal reset
    wire rst = ~ARESETn;

    // =========================================================
    //  CSR Registers (kept in this module)
    // =========================================================
    reg [15:0]  n_objects_reg;
    reg signed [7:0]  task_emb_reg  [0:EMB_DIM-1];

    reg         done_reg;
    reg         busy_reg;
    reg [15:0]  best_object_reg;
    reg [31:0]  best_score_reg;
    reg         start_pulse;       // 1-cycle start strobe

    // =========================================================
    //  Word-address constants (CSR region, byte_addr >> 2)
    // =========================================================
    localparam W_CTRL          = 10'd0;
    localparam W_STATUS        = 10'd1;
    localparam W_N_OBJ         = 10'd2;
    localparam W_BEST_OBJ      = 10'd3;
    localparam W_BEST_SCORE    = 10'd4;
    localparam W_TASK_EMB_BASE = 10'd64;   // 0x100 >> 2
    localparam W_CONF_BASE     = 10'd128;  // 0x200 >> 2
    localparam W_BOOST_BASE    = 10'd192;  // 0x300 >> 2
    localparam W_PENALTY_BASE  = 10'd256;  // 0x400 >> 2

    // =========================================================
    //  Task embedding flat bus (shared by all lanes)
    // =========================================================
    wire signed [8*EMB_DIM-1:0] task_emb_flat;
    genvar tgi;
    generate
        for (tgi = 0; tgi < EMB_DIM; tgi = tgi + 1) begin : gen_task_flat
            assign task_emb_flat[8*tgi +: 8] = task_emb_reg[tgi];
        end
    endgenerate

    // =========================================================
    //  Scorer ↔ AXI write-path signals
    // =========================================================
    reg  [EMB_DIM-1:0]           scorer_emb_wr_we;
    reg  [OBJ_IDX_W-1:0]        scorer_emb_wr_addr;
    reg  [8*EMB_DIM-1:0]        scorer_emb_wr_din;

    reg                          scorer_meta_conf_we;
    reg                          scorer_meta_boost_we;
    reg                          scorer_meta_penalty_we;
    reg  [OBJ_IDX_W-1:0]        scorer_meta_wr_addr;
    reg  signed [15:0]           scorer_meta_wr_conf;
    reg  signed [15:0]           scorer_meta_wr_boost;
    reg                          scorer_meta_wr_penalty;

    // =========================================================
    //  Scorer ↔ AXI read-path signals
    // =========================================================
    reg  [OBJ_IDX_W-1:0]        scorer_emb_axi_rd_addr;
    wire [8*EMB_DIM-1:0]        scorer_emb_axi_rd_dout;

    reg  [OBJ_IDX_W-1:0]        scorer_meta_rd_addr;
    wire signed [15:0]           scorer_meta_rd_conf;
    wire signed [15:0]           scorer_meta_rd_boost;
    wire                         scorer_meta_rd_penalty;

    // =========================================================
    //  Scorer results
    // =========================================================
    wire        scorer_done;
    wire [15:0] scorer_best_object;
    wire signed [31:0] scorer_best_score;

    // =========================================================
    //  Parallel Scorer Instance
    // =========================================================
    parallel_scorer #(
        .EMB_DIM    (EMB_DIM),
        .MAX_OBJECTS(MAX_OBJECTS),
        .N_LANES    (N_LANES)
    ) u_scorer (
        .clk              (ACLK),
        .rst              (rst),

        .start            (start_pulse),
        .n_objects         (n_objects_reg),
        .task_emb_flat     (task_emb_flat),

        // BRAM write (broadcast)
        .emb_wr_we         (scorer_emb_wr_we),
        .emb_wr_addr       (scorer_emb_wr_addr),
        .emb_wr_din        (scorer_emb_wr_din),

        // BRAM read (AXI readback)
        .emb_axi_rd_addr   (scorer_emb_axi_rd_addr),
        .emb_axi_rd_dout   (scorer_emb_axi_rd_dout),

        // Metadata write
        .meta_conf_we      (scorer_meta_conf_we),
        .meta_boost_we     (scorer_meta_boost_we),
        .meta_penalty_we   (scorer_meta_penalty_we),
        .meta_wr_addr      (scorer_meta_wr_addr),
        .meta_wr_conf      (scorer_meta_wr_conf),
        .meta_wr_boost     (scorer_meta_wr_boost),
        .meta_wr_penalty   (scorer_meta_wr_penalty),

        // Metadata read (AXI readback)
        .meta_rd_addr      (scorer_meta_rd_addr),
        .meta_rd_conf      (scorer_meta_rd_conf),
        .meta_rd_boost     (scorer_meta_rd_boost),
        .meta_rd_penalty   (scorer_meta_rd_penalty),

        // Results
        .done              (scorer_done),
        .best_object       (scorer_best_object),
        .best_score        (scorer_best_score)
    );

    // =========================================================
    //  Done / Busy tracking
    // =========================================================
    always @(posedge ACLK or negedge ARESETn) begin
        if (!ARESETn) begin
            done_reg        <= 1'b0;
            busy_reg        <= 1'b0;
            best_object_reg <= 16'h0;
            best_score_reg  <= 32'h0;
        end else begin
            if (start_pulse) begin
                done_reg <= 1'b0;
                busy_reg <= 1'b1;
            end
            if (scorer_done) begin
                done_reg        <= 1'b1;
                busy_reg        <= 1'b0;
                best_object_reg <= scorer_best_object;
                best_score_reg  <= scorer_best_score;
            end
        end
    end

    // =========================================================
    //  AXI Write Path
    // =========================================================
    localparam AW_IDLE = 2'd0, AW_DATA = 2'd1, AW_RESP = 2'd2;
    reg [1:0]             aw_state;
    reg [AXI_ID_W-1:0]   aw_id_r;
    reg [AXI_ADDR_W-1:0] aw_addr_r;
    reg [7:0]             aw_len_r;
    reg [7:0]             aw_beat;

    // Address decode wires (from registered aw_addr_r)
    wire [15:0] wa          = aw_addr_r[15:0];
    wire        wa_csr      = (wa[15:12] == 4'h0);
    wire        wa_obj_emb  = (wa[15:12] == 4'h1);
    wire [9:0]  wa_word     = wa[11:2];
    wire [EMB_DIM_LOG2-1:0] wa_byte_idx = wa[2 +: EMB_DIM_LOG2];
    wire [OBJ_IDX_W-1:0]    wa_obj_id   = wa[2 + EMB_DIM_LOG2 +: OBJ_IDX_W];

    integer jj;

    always @(posedge ACLK or negedge ARESETn) begin : axi_wr
        if (!ARESETn) begin
            aw_state  <= AW_IDLE;
            AWREADY   <= 1'b0;
            WREADY    <= 1'b0;
            BVALID    <= 1'b0;
            BID       <= {AXI_ID_W{1'b0}};
            BRESP     <= 2'b00;
            aw_addr_r <= {AXI_ADDR_W{1'b0}};
            aw_len_r  <= 8'h0;
            aw_beat   <= 8'h0;
            start_pulse         <= 1'b0;
            n_objects_reg       <= 16'h0;
            scorer_emb_wr_we    <= {EMB_DIM{1'b0}};
            scorer_emb_wr_addr  <= {OBJ_IDX_W{1'b0}};
            scorer_emb_wr_din   <= {(8*EMB_DIM){1'b0}};
            scorer_meta_conf_we    <= 1'b0;
            scorer_meta_boost_we   <= 1'b0;
            scorer_meta_penalty_we <= 1'b0;
            scorer_meta_wr_addr    <= {OBJ_IDX_W{1'b0}};
            scorer_meta_wr_conf    <= 16'sd0;
            scorer_meta_wr_boost   <= 16'sd0;
            scorer_meta_wr_penalty <= 1'b0;
            for (jj = 0; jj < EMB_DIM; jj = jj + 1)
                task_emb_reg[jj] <= 8'sd0;
        end else begin
            // ── Default de-assertions ────────────────────────
            start_pulse            <= 1'b0;
            scorer_emb_wr_we       <= {EMB_DIM{1'b0}};
            scorer_meta_conf_we    <= 1'b0;
            scorer_meta_boost_we   <= 1'b0;
            scorer_meta_penalty_we <= 1'b0;

            case (aw_state)
                // ── Wait for write address ────────────────────
                AW_IDLE: begin
                    AWREADY <= 1'b1;
                    WREADY  <= 1'b0;
                    BVALID  <= 1'b0;
                    if (AWVALID && AWREADY) begin
                        aw_id_r   <= AWID;
                        aw_addr_r <= AWADDR;
                        aw_len_r  <= AWLEN;
                        aw_beat   <= 8'h0;
                        AWREADY   <= 1'b0;
                        WREADY    <= 1'b1;
                        aw_state  <= AW_DATA;
                    end
                end

                // ── Accept write data beats ───────────────────
                AW_DATA: begin
                    if (WVALID && WREADY) begin
                        // ─ CSR region (0x0000–0x0FFF) ─
                        if (wa_csr) begin
                            if (wa_word == W_CTRL) begin
                                if (WDATA[0]) start_pulse <= 1'b1;
                            end else if (wa_word == W_N_OBJ) begin
                                n_objects_reg <= WDATA[15:0];
                            end else if (wa_word >= W_TASK_EMB_BASE &&
                                         wa_word <  W_TASK_EMB_BASE + EMB_DIM) begin
                                task_emb_reg[wa_word - W_TASK_EMB_BASE]
                                    <= $signed(WDATA[7:0]);
                            end else if (wa_word >= W_CONF_BASE &&
                                         wa_word <  W_CONF_BASE + MAX_OBJECTS) begin
                                scorer_meta_conf_we <= 1'b1;
                                scorer_meta_wr_addr <= wa_word[OBJ_IDX_W-1:0]
                                                       - W_CONF_BASE[OBJ_IDX_W-1:0];
                                scorer_meta_wr_conf <= $signed(WDATA[15:0]);
                            end else if (wa_word >= W_BOOST_BASE &&
                                         wa_word <  W_BOOST_BASE + MAX_OBJECTS) begin
                                scorer_meta_boost_we <= 1'b1;
                                scorer_meta_wr_addr  <= wa_word[OBJ_IDX_W-1:0]
                                                        - W_BOOST_BASE[OBJ_IDX_W-1:0];
                                scorer_meta_wr_boost <= $signed(WDATA[15:0]);
                            end else if (wa_word >= W_PENALTY_BASE &&
                                         wa_word <  W_PENALTY_BASE + MAX_OBJECTS) begin
                                scorer_meta_penalty_we <= 1'b1;
                                scorer_meta_wr_addr    <= wa_word[OBJ_IDX_W-1:0]
                                                          - W_PENALTY_BASE[OBJ_IDX_W-1:0];
                                scorer_meta_wr_penalty <= WDATA[0];
                            end
                        end
                        // ─ OBJ_EMB region (0x1000–0x1FFF) → BRAM ─
                        else if (wa_obj_emb) begin
                            if (wa_obj_id < MAX_OBJECTS) begin
                                scorer_emb_wr_we   <= ({{(EMB_DIM-1){1'b0}}, 1'b1}
                                                       << wa_byte_idx);
                                scorer_emb_wr_addr <= wa_obj_id;
                                scorer_emb_wr_din  <= {EMB_DIM{WDATA[7:0]}};
                            end
                        end

                        // Burst: increment address (INCR mode)
                        aw_addr_r <= aw_addr_r + 4;
                        aw_beat   <= aw_beat + 1;

                        if (WLAST) begin
                            WREADY   <= 1'b0;
                            BVALID   <= 1'b1;
                            BID      <= aw_id_r;
                            BRESP    <= 2'b00;  // OKAY
                            aw_state <= AW_RESP;
                        end
                    end
                end

                // ── Send write response ───────────────────────
                AW_RESP: begin
                    if (BVALID && BREADY) begin
                        BVALID   <= 1'b0;
                        aw_state <= AW_IDLE;
                    end
                end

                default: aw_state <= AW_IDLE;
            endcase
        end
    end

    // =========================================================
    //  AXI Read Path  (with 1-cycle BRAM pipeline stage)
    // =========================================================
    localparam AR_IDLE = 2'd0, AR_PIPE = 2'd1, AR_DATA = 2'd2;
    reg [1:0]            ar_state;
    reg [AXI_ID_W-1:0]  ar_id_r;
    reg [AXI_ADDR_W-1:0] ar_addr_r;
    reg [7:0]            ar_len_r;
    reg [7:0]            ar_beat;

    // Address decode wires (from registered ar_addr_r)
    wire [15:0] ra          = ar_addr_r[15:0];
    wire        ra_csr      = (ra[15:12] == 4'h0);
    wire        ra_obj_emb  = (ra[15:12] == 4'h1);
    wire [9:0]  ra_word     = ra[11:2];
    wire [EMB_DIM_LOG2-1:0] ra_byte_idx = ra[2 +: EMB_DIM_LOG2];
    wire [OBJ_IDX_W-1:0]    ra_obj_id   = ra[2 + EMB_DIM_LOG2 +: OBJ_IDX_W];

    // ── BRAM read address (continuously driven) ──────────────
    always @(*) begin
        scorer_emb_axi_rd_addr = ra_obj_id;
        scorer_meta_rd_addr    = {OBJ_IDX_W{1'b0}};

        if (ra_csr) begin
            if (ra_word >= W_CONF_BASE && ra_word < W_CONF_BASE + MAX_OBJECTS)
                scorer_meta_rd_addr = ra_word[OBJ_IDX_W-1:0]
                                      - W_CONF_BASE[OBJ_IDX_W-1:0];
            else if (ra_word >= W_BOOST_BASE && ra_word < W_BOOST_BASE + MAX_OBJECTS)
                scorer_meta_rd_addr = ra_word[OBJ_IDX_W-1:0]
                                      - W_BOOST_BASE[OBJ_IDX_W-1:0];
            else if (ra_word >= W_PENALTY_BASE && ra_word < W_PENALTY_BASE + MAX_OBJECTS)
                scorer_meta_rd_addr = ra_word[OBJ_IDX_W-1:0]
                                      - W_PENALTY_BASE[OBJ_IDX_W-1:0];
        end
    end

    // ── Combinational read-data mux ──────────────────────────
    //    For CSRs + metadata: combinational (0-cycle).
    //    For OBJ_EMB BRAM: uses registered BRAM output (via AR_PIPE).
    wire [7:0] emb_rd_byte = scorer_emb_axi_rd_dout[ra_byte_idx*8 +: 8];

    reg [AXI_DATA_W-1:0] rd_mux;
    always @(*) begin
        rd_mux = 32'hDEAD_BEEF;
        if (ra_csr) begin
            if      (ra_word == W_CTRL)       rd_mux = 32'h0;
            else if (ra_word == W_STATUS)     rd_mux = {30'h0, busy_reg, done_reg};
            else if (ra_word == W_N_OBJ)      rd_mux = {16'h0, n_objects_reg};
            else if (ra_word == W_BEST_OBJ)   rd_mux = {16'h0, best_object_reg};
            else if (ra_word == W_BEST_SCORE) rd_mux = best_score_reg;
            else if (ra_word >= W_TASK_EMB_BASE &&
                     ra_word <  W_TASK_EMB_BASE + EMB_DIM)
                rd_mux = {{24{task_emb_reg[ra_word - W_TASK_EMB_BASE][7]}},
                              task_emb_reg[ra_word - W_TASK_EMB_BASE]};
            else if (ra_word >= W_CONF_BASE &&
                     ra_word <  W_CONF_BASE + MAX_OBJECTS)
                rd_mux = {{16{scorer_meta_rd_conf[15]}}, scorer_meta_rd_conf};
            else if (ra_word >= W_BOOST_BASE &&
                     ra_word <  W_BOOST_BASE + MAX_OBJECTS)
                rd_mux = {{16{scorer_meta_rd_boost[15]}}, scorer_meta_rd_boost};
            else if (ra_word >= W_PENALTY_BASE &&
                     ra_word <  W_PENALTY_BASE + MAX_OBJECTS)
                rd_mux = {31'h0, scorer_meta_rd_penalty};
            else
                rd_mux = 32'h0;
        end else if (ra_obj_emb) begin
            if (ra_obj_id < MAX_OBJECTS)
                rd_mux = {{24{emb_rd_byte[7]}}, emb_rd_byte};
        end
    end

    // ── AXI Read FSM ─────────────────────────────────────────
    always @(posedge ACLK or negedge ARESETn) begin : axi_rd
        if (!ARESETn) begin
            ar_state  <= AR_IDLE;
            ARREADY   <= 1'b0;
            RVALID    <= 1'b0;
            RLAST     <= 1'b0;
            RID       <= {AXI_ID_W{1'b0}};
            RDATA     <= 32'h0;
            RRESP     <= 2'b00;
            ar_addr_r <= {AXI_ADDR_W{1'b0}};
            ar_len_r  <= 8'h0;
            ar_beat   <= 8'h0;
        end else begin
            case (ar_state)
                AR_IDLE: begin
                    ARREADY <= 1'b1;
                    RVALID  <= 1'b0;
                    RLAST   <= 1'b0;
                    if (ARVALID && ARREADY) begin
                        ar_id_r   <= ARID;
                        ar_addr_r <= ARADDR;
                        ar_len_r  <= ARLEN;
                        ar_beat   <= 8'h0;
                        ARREADY   <= 1'b0;
                        ar_state  <= AR_PIPE;   // BRAM pipeline stage
                    end
                end

                // ── Pipeline: 1 cycle for BRAM output to register ─
                AR_PIPE: begin
                    ar_state <= AR_DATA;
                end

                AR_DATA: begin
                    RDATA  <= rd_mux;
                    RID    <= ar_id_r;
                    RRESP  <= 2'b00;
                    RVALID <= 1'b1;
                    RLAST  <= (ar_beat == ar_len_r);

                    if (RVALID && RREADY) begin
                        if (ar_beat == ar_len_r) begin
                            RVALID   <= 1'b0;
                            RLAST    <= 1'b0;
                            ARREADY  <= 1'b1;
                            ar_state <= AR_IDLE;
                        end else begin
                            ar_addr_r <= ar_addr_r + 4;
                            ar_beat   <= ar_beat + 1;
                            RVALID    <= 1'b0;
                            ar_state  <= AR_PIPE;  // re-pipeline for next beat
                        end
                    end
                end

                default: ar_state <= AR_IDLE;
            endcase
        end
    end

endmodule
