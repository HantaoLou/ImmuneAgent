/**
 * 工具 Demo 数据配置
 * 为每个服务的每个工具提供可执行的 demo 数据
 * 格式: { service_id: { tool_name: { param_name: value } } }
 */

export interface ToolDemoData {
  [serviceId: string]: {
    [toolName: string]: Record<string, any>
  }
}

export const toolDemoData: ToolDemoData = {
  airr: {
    search_airr_repertoires: {
      disease: 'COVID-19',
      tissue: 'peripheral blood',
      species: 'human',
      cell_subset: 'memory',
      repository: 'all',
      max_results: 10,
    },
    download_airr_sequences: {
      repertoire_id: '5ed6859e99011334ac05e847',
      filters: null,
      format: 'airr',
      max_sequences: 1000,
      output_dir: '/data_new/workspace/airr_result',
    },
    get_airr_study_metadata: {
      study_id: 'PRJNA300878',
      repository: 'auto',
    },
    filter_by_vdj_genes: {
      repertoire_id: '5ed6859e99011334ac05e847',
      v_gene: 'IGHV3',
      d_gene: null,
      j_gene: null,
      combination_logic: 'AND',
    },
    get_airr_statistics: {
      repertoire_id: '5ed6859e99011334ac05e847',
      metrics: [],
    },
  },
  anarci: {
    number_single_sequence: {
      sequence: 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWINWVRQAPGQGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
      scheme: 'imgt',
    },
    number_antibody_batch: {
      sequences: JSON.stringify([
        {
          id: 'seq1',
          sequence: 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWINWVRQAPGQGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
          chain_type: 'heavy',
        },
        {
          id: 'seq2',
          sequence: 'DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTLPWTFGGGTKVEIK',
          chain_type: 'light',
        },
        {
          id: 'seq3',
          sequence: 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDYGDYFDYWGQGTLVTVSS',
          chain_type: 'heavy',
        },
        {
          id: 'seq4',
          sequence: 'EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPPTFGQGTKVEIK',
          chain_type: 'light',
        },
        {
          id: 'seq5',
          sequence: 'QVQLQQSGAELARPGASVKMSCKASGYTFTSYWINWVRQAPGQGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
          chain_type: 'heavy',
        },
        {
          id: 'seq6',
          sequence: 'DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTLPWTFGGGTKVEIK',
          chain_type: 'light',
        },
        {
          id: 'seq7',
          sequence: 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDYGDYFDYWGQGTLVTVSS',
          chain_type: 'heavy',
        },
        {
          id: 'seq8',
          sequence: 'EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPPTFGQGTKVEIK',
          chain_type: 'light',
        },
        {
          id: 'seq9',
          sequence: 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWINWVRQAPGQGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
          chain_type: 'heavy',
        },
        {
          id: 'seq10',
          sequence: 'DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTLPWTFGGGTKVEIK',
          chain_type: 'light',
        },
        {
          id: 'seq11',
          sequence: 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDYGDYFDYWGQGTLVTVSS',
          chain_type: 'heavy',
        },
        {
          id: 'seq12',
          sequence: 'EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQRSNWPPTFGQGTKVEIK',
          chain_type: 'light',
        },
      ]),
      scheme: 'imgt',
      assign_germline: true,
    },
  },
  sabdab: {
    download_sabdab_summary: {
      output_path: '/data_new/workspace/sabdab_result',
      numbering_scheme: 'imgt',
      include_pdb: true,
    },
    download_sabdab_structure: {
      pdb_id: '1HZH',
      output_path: '/data_new/workspace/sabdab_result',
    },
  },
  lgblast: {
    analyze_vdj_batch: {
      sequences: '/data_new/workspace/antibody_gen/mcp_Igblast/igblast_changeO/input/rsvH.fasta',
      organism: 'human',
      receptor_type: 'Ig',
      locus: 'IGH',
    },
    extract_cdr3_from_airr: {
      airr_results: [],
    },
  },
  metabcr: {
    metabcr: {
      input_file_path: '/data_new/workspace/AgeB_BCR_standardized.csv',
      output_file_path: '/data_new/workspace/metabcr_result',
    },
  },
  af3: {
    alphafold3: {
      input_file_path: '/data_new/workspace/20250401_AF3.xlsx',
      antigen_name: 'H5N1_TEXAS',
    },
  },
  fdg: {
    fdg: {
      input_file_path: '/data_new/workspace/nk1_61_model.pdb',
    },
  },
  bcell: {
    analyze_affinity_maturation: {
      airr_file: '/path/to/airr.tsv',
      input_rds: '/path/to/input.rds',
      pseudotime_column: 'pseudotime',
    },
    analyze_bcell_trajectory: {
      input_rds: '/path/to/input.rds',
      method: 'monocle3',
      root_state: 'Naive',
      use_partition: true,
    },
    analyze_clonal_expansion: {
      airr_file: '/path/to/airr.tsv',
      subset_metadata: 'celltype',
      top_n_clones: 150,
    },
    analyze_gc_dynamics: {
      gc_subsets: ['DZ_GC', 'LZ_GC'],
      input_rds: '/path/to/input.rds',
      plot_markers: true,
    },
    analyze_isotype_distribution: {
      bcr_data: '/path/to/bcr.tsv',
      input_rds: '/path/to/input.rds',
      plot_type: 'bar',
    },
    analyze_vdj_usage: {
      airr_file: '/path/to/airr.tsv',
      plot_heatmap: true,
      subset_metadata: 'celltype',
    },
    calculate_repertoire_diversity: {
      airr_file: '/path/to/airr.tsv',
      grouping_column: 'subset',
      metrics: ['shannon', 'simpson', 'hill'],
    },
    calculate_shm_rates: {
      airr_file: '/path/to/airr.tsv',
      input_rds: '/path/to/input.rds',
      min_sequences: 10,
      region: 'IGHV',
    },
    compare_rsv_a_vs_b_bcells: {
      input_rds: '/path/to/input.rds',
      logfc_threshold: 0.25,
      min_pct: 0.1,
      rsv_metadata: 'rsv_type',
    },
    generate_bcell_report: {
      airr_file: '/path/to/airr.tsv',
      cytotrace_rds: '/path/to/cytotrace.rds',
      input_rds: '/path/to/input.rds',
      report_format: 'html',
      trajectory_rds: '/path/to/trajectory.rds',
    },
    identify_antigen_specific_bcells: {
      binding_predictions: '/path/to/binding.csv',
      input_rds: '/path/to/input.rds',
      min_binding_score: 0.5,
    },
    identify_bcell_subsets: {
      input_rds: '/path/to/input.rds',
      resolution: 0.8,
      species: 'human',
    },
    identify_convergent_sequences: {
      airr_file: '/path/to/airr.tsv',
      cdr3_similarity_threshold: 0.85,
      min_group_size: 2,
    },
    predict_plasma_potential: {
      gene_signatures: '/path/to/signatures.csv',
      input_rds: '/path/to/input.rds',
    },
    run_cytotrace_analysis: {
      enable_fast: true,
      input_rds: '/path/to/input.rds',
      ncores: 4,
      subsample_size: 1000,
    },
    run_figure2_deg_analysis: {
      flu_data_path: '/path/to/flu_data.rds',
      sars_data_path: '/path/to/sars_data.rds',
      rsv_data_path: '/path/to/rsv_data.rds',
      flu_binding_threshold: 0.625,
      sars_binding_threshold: 0.5,
      rsv_binding_threshold: 0.5,
      output_dir: '/path/to/output',
    },
    run_figure3_correlation_analysis: {
      figure2_results_dir: '/path/to/figure2_results',
      output_dir: '/path/to/output',
    },
    run_figure4_trajectory_analysis: {
      a1a11_data_path: '/path/to/a1a11_data.rds',
      flu_data_path: '/path/to/flu_data.rds',
      num_dim: 50,
      k_neighbors: 40,
      resolution: 0.001,
      output_dir: '/path/to/output',
    },
    run_figure5_bcr_analysis: {
      flu_data_path: '/path/to/flu_data.rds',
      a1a11_data_path: '/path/to/a1a11_data.rds',
      bcr_file_path: '/path/to/bcr_file.rds',
      shm_file_path: '/path/to/shm_file.rds',
      shm_outlier_cutoff: 45,
      output_dir: '/path/to/output',
    },
  },
  bioinformatics: {
    antigen_binding_neutralization_density_visualization: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      feature_priority: 'neutralization_first',
      input_file: '/data_new/workspace/Age_Bcells.rds',
      na_strategy: 'exclude_cells',
      prediction_keywords: 'neut,bind,average,predict,output',
    },
    antigen_binding_prediction_visualization: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      binding_threshold: 0.5,
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    bcell_celltype_distribution_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    bcell_celltype_umap_visualization: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      celltype_column: 'CellType',
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    bcell_marker_gene_dotplot_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      input_file: '/data_new/workspace/Age_Bcells.rds',
      min_expression: 0.25,
      min_pct: 0.1,
    },
    bcell_marker_gene_expression_dotplot: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      celltype_column: 'CellType',
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    bcr_isotype_distribution_shm_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      binding_threshold: 0.5,
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    binding_prediction_interval_distribution_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      data_max: 1.0,
      data_min: 0.0,
      input_file: '/data_new/workspace/Age_Bcells.rds',
      interval_step: 0.1,
    },
    differential_gene_correlation_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      dataset1_name: 'Dataset1',
      dataset2_name: 'Dataset2',
      deg_file1: '/data_new/workspace/deg1.csv',
      deg_file2: '/data_new/workspace/deg2.csv',
      highlight_genes: 'ITGAX,FGR,FCRL4,FCRL5',
      min_common_genes: 10,
      p_value_threshold: 0.05,
    },
    differential_gene_expression_volcano_analysis: {
      analysis_strategy: 'both',
      base_dir: '/data_new/workspace/bioinformatics_result',
      input_file: '/data_new/workspace/Age_Bcells.rds',
      logfc_threshold: 0.0,
      min_pct: 0.2,
    },
    neutralizing_antibody_shm_comparison_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      binding_threshold: 0.5,
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    prediction_value_density_visualization: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      input_file: '/data_new/workspace/Age_Bcells.rds',
      prediction_keywords: 'bind,predict,output,average,score',
      prediction_threshold: 0.5,
    },
    pseudotime_celltype_boxplot_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      celltype_column: '',
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    pseudotime_trajectory_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      cluster_resolution: 0.001,
      input_file: '/data_new/workspace/Age_Bcells.rds',
      min_gene_cells: 3,
      num_dim: 50,
      root_celltype: 'Naive',
    },
    trajectory_polynomial_regression_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    trajectory_supplementary_analysis: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
    umap_dimensionality_reduction_visualization: {
      base_dir: '/data_new/workspace/bioinformatics_result',
      input_file: '/data_new/workspace/Age_Bcells.rds',
    },
  },
  annotation: {
    annotate_by_markers: {
      cluster_column: 'seurat_clusters',
      input_rds: '/path/to/input.rds',
      new_column: 'manual_celltype',
    },
    detect_cluster_markers: {
      input_rds: '/data_new/workspace/Age_Bcells.rds',
      logfc_threshold: 0.5,
      min_pct: 0.25,
      only_pos: true,
      test_use: 'wilcox',
      top_n: 10,
    },
    export_annotations: {
      annotation_columns: [],
      export_format: 'csv',
      include_umap: true,
      input_rds: '/data_new/workspace/Age_Bcells.rds',
    },
    run_singler_annotation: {
      cluster_column: 'seurat_clusters',
      input_rds: '/data_new/workspace/Age_Bcells.rds',
      label_type: 'label.main',
      reference_dataset: 'HumanPrimaryCellAtlasData',
    },
    score_annotation_confidence: {
      annotation_column: 'celltype',
      input_rds: '/data_new/workspace/Age_Bcells.rds',
    },
    validate_annotation: {
      annotation_column1: 'celltype1',
      annotation_column2: 'celltype2',
      input_rds: '/data_new/workspace/Age_Bcells.rds',
      reference_dataset: 'MonacoImmuneData',
    },
  },
  geo: {
    download_geo_sequences: {
      data_format: 'processed',
      geo_id: 'GSE123456',
      max_sequences: 10000,
      output_format: 'fasta',
    },
    download_iedb_dataset: {
      antigen: 'H5N1_TEXAS',
      assay_type: 'all',
      disease: 'COVID-19',
      format: 'json',
      max_records: 10000,
    },
    enrich_sequences_with_iedb: {
      iedb_search: true,
      sabdab_search: false,
      sequences: [],
    },
    get_all_geo_bcr_studies: {
      include_metadata: true,
      max_results: 1000,
      organism: 'Homo sapiens',
    },
    get_geo_metadata: {
      geo_id: 'GSE123456',
      include_samples: true,
    },
    get_iedb_antibody_data: {
      antibody_id: 'AB123456',
      antibody_name: 'example_antibody',
      include_structures: true,
    },
    list_geo_bcr_datasets: {
      curated_only: true,
    },
    map_cdr3_to_epitopes: {
      cdr3_list: ['CARDRYYYGMDV', 'CARDYYYGMDV', 'CARDYYYGM'],
      confidence_threshold: 0.7,
      disease_context: 'COVID-19',
    },
    search_geo_studies: {
      keywords: 'BCR sequencing',
      max_results: 100,
      organism: 'Homo sapiens',
      study_type: 'BCR-seq',
    },
    search_iedb_epitopes: {
      antibody_sequence: 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWINWVRQAPGQGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
      antigen: 'H5N1_TEXAS',
      assay_type: 'binding',
      disease: 'COVID-19',
      max_results: 1000,
      organism: 'Homo sapiens',
    },
    validate_epitope_binding: {
      antigen: 'H5N1_TEXAS',
      cdr3_sequences: JSON.stringify(['CARGLVVVADAFDIW', 'CARDRGWGFEHFDYW', 'CARYYDSSGYNWFDPW']),
      epitope_sequence: 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWINWVRQAPGQGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
    },
  },
  oas: {
    batch_process_oas: {
      batch_size: 100000,
      operation: 'download',
      parallel: false,
      study_ids: JSON.stringify(['PRJNA300878', 'PRJNA300879', 'PRJNA300880']),
    },
    clear_cache: {
      keep_recent: 0,
    },
    download_oas_dataset: {
      cache: true,
      data_units: [],
      format: 'airr',
      max_sequences: 500000,
      study_id: 'PRJNA300878',
    },
    filter_oas_by_vgene: {
      cdr3_pattern: 'CAR.*',
      j_gene: 'IGHJ4',
      min_identity: 0.0,
      study_id: 'PRJNA300878',
      v_gene: 'IGHV3',
    },
    get_all_oas_studies: {
      force_refresh: false,
      include_metadata: true,
    },
    get_oas_summary: {
      study_id: 'PRJNA300878',
    },
    search_oas_studies: {
      disease: 'COVID-19',
      min_sequences: 5000000,
      paired: false,
      species: 'human',
      tissue: 'peripheral blood',
    },
  },
  communication: {
    analyze_signaling_pathways: {
      cellchat_rds: '/path/to/cellchat.rds',
      output_prefix: 'pathways',
      pathways: 'all',
    },
    calculate_network_centrality: {
      cellchat_rds: '/path/to/cellchat.rds',
      output_prefix: 'centrality',
    },
    compare_communication_networks: {
      cellchat_list: '/path/to/cellchat_list.rds',
      group_names: 'group1,group2',
      output_prefix: 'comparison',
    },
    identify_ligand_receptor_pairs: {
      group_by: 'celltype',
      input_rds: '/path/to/input.rds',
      min_pct: 0.1,
      output_prefix: 'lr_pairs',
      species: 'mouse',
    },
    plot_communication_network: {
      cellchat_rds: '/path/to/cellchat.rds',
      layout: 'circle',
      output_prefix: 'network',
      signaling: 'TGFb',
    },
    run_cellchat_analysis: {
      db_type: 'Secreted Signaling',
      group_by: 'celltype',
      input_rds: '/path/to/input.rds',
      output_prefix: 'cellchat',
      species: 'mouse',
    },
    run_nichenet_analysis: {
      condition_colname: 'orig.ident',
      condition_oi: 'treatment',
      condition_reference: 'control',
      input_rds: '/path/to/input.rds',
      output_prefix: 'nichenet',
      receiver_cells: 'B_cell',
      sender_cells: 'T_cell',
    },
    run_spatial_communication: {
      contact_range: 100.0,
      group_by: 'celltype',
      input_rds: '/path/to/input.rds',
      output_prefix: 'spatial_comm',
      spatial_coordinates: 'x,y',
      species: 'mouse',
    },
  },
  multimodal: {
    identify_multimodal_clusters: {
      algorithm: 3,
      resolution: 0.8,
      wnn_rds: '/path/to/wnn.rds',
    },
    integrate_multimodal_wnn: {
      adt_dims: 18,
      adt_rds: '/path/to/adt.rds',
      atac_dims: 30,
      atac_rds: '/path/to/atac.rds',
      rna_dims: 30,
      rna_rds: '/path/to/rna.rds',
    },
    link_peaks_to_genes: {
      atac_rds: '/path/to/atac.rds',
      distance: 500000,
      min_correlation: 0.05,
      rna_rds: '/path/to/rna.rds',
    },
    plot_multimodal_features: {
      features: ['CD3', 'CD4', 'CD8'],
      modalities: ['rna', 'adt'],
      plot_type: 'umap',
      wnn_rds: '/path/to/wnn.rds',
    },
    process_adt_data: {
      adt_counts_file: '/path/to/adt_counts.h5',
      min_cells: 3,
      min_features: 3,
      normalization_method: 'CLR',
    },
    process_atac_data: {
      fragments_file: '/path/to/fragments.tsv.gz',
      genome: 'hg38',
      max_fragments: 100000,
      min_cells: 10,
      min_fragments: 1000,
      peaks_file: '/path/to/peaks.bed',
      tss_enrichment_threshold: 2.0,
    },
  },
  /**
   * scrna: Single-cell RNA sequencing (scRNA-seq) analysis service
   * 
   * Provides comprehensive single-cell RNA-seq data analysis tools including:
   * - Quality control and preprocessing (normalization, filtering, variable feature selection)
   * - Dimensionality reduction (PCA, UMAP, t-SNE)
   * - Clustering analysis (Leiden, Louvain algorithms)
   * - Batch integration and correction (Harmony)
   * - Doublet detection and removal
   * - Differential expression analysis (DEG)
   * - Marker gene detection and identification
   * - Pathway enrichment analysis
   * 
   * Bioinformatics domains: ["single-cell", "scRNA-seq", "transcriptomics", "clustering", "differential expression"]
   * Input data: ["Single-cell RNA-seq RDS files", "Seurat objects", "Expression matrices"]
   * Output results: ["Clustered data", "DEG results", "Visualization plots", "Marker genes", "Pathway analysis"]
   */
  scrna: {
    run_clustering_analysis: {
      algorithm: 'leiden',
      dims: 30,
      input_rds: '/path/to/input.rds',
      resolution: 0.5,
    },
    run_deg_analysis: {
      group_by: 'seurat_clusters',
      ident_1: 'cluster1',
      ident_2: 'cluster2',
      input_rds: '/path/to/input.rds',
      logfc_threshold: 0.25,
      min_pct: 0.1,
      test_use: 'wilcox',
    },
    run_dim_reduction: {
      dims: 30,
      input_rds: '/path/to/input.rds',
      methods: ['pca', 'umap'],
      min_dist: 0.3,
      n_neighbors: 30,
    },
    run_doublet_detection: {
      dims: 20,
      expected_doublet_rate: 0.08,
      input_rds: '/path/to/input.rds',
      pK: 0.09,
      pN: 0.25,
    },
    run_full_preprocessing_pipeline: {
      dims: 30,
      input_rds: '/path/to/input.rds',
      max_genes: 10000,
      min_counts: 1000,
      min_genes: 200,
      mt_percent: 25.0,
      n_variable_features: 3000,
      resolution: 0.8,
      vars_to_regress: [],
    },
    run_integration_harmony: {
      batch_variable: 'orig.ident',
      dims: 30,
      input_rds: '/path/to/input.rds',
      theta: [],
    },
    run_marker_detection: {
      group_by: 'seurat_clusters',
      input_rds: '/path/to/input.rds',
      logfc_threshold: 0.5,
      min_pct: 0.25,
      only_pos: true,
      top_n: 10,
    },
    run_normalization_sct: {
      input_rds: '/path/to/input.rds',
      n_variable_features: 3000,
      vars_to_regress: [],
    },
    run_pathway_enrichment: {
      deg_csv: '/path/to/deg.csv',
      input_rds: '/path/to/input.rds',
      ontology: 'BP',
      organism: 'human',
      pvalue_cutoff: 0.05,
      qvalue_cutoff: 0.2,
    },
    run_qc_filtering: {
      input_rds: '/path/to/input.rds',
      max_genes: 6000,
      min_counts: 1000,
      min_genes: 200,
      mt_percent: 20.0,
    },
    run_subset_cells: {
      input_rds: '/path/to/input.rds',
      invert: false,
      subset_column: 'celltype',
      subset_values: ['B_cell', 'T_cell', 'NK_cell'],
    },
  },
}

/**
 * 获取指定服务和工具的 demo 数据
 */
export function getToolDemoData(serviceId: string, toolName: string): Record<string, any> | null {
  const serviceData = toolDemoData[serviceId]
  if (!serviceData) {
    return null
  }
  return serviceData[toolName] || null
}

/**
 * 检查指定服务和工具是否有 demo 数据
 */
export function hasToolDemoData(serviceId: string, toolName: string): boolean {
  return getToolDemoData(serviceId, toolName) !== null
}

