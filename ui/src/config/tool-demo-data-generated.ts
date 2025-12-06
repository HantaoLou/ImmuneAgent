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
  af3: {
    alphafold3: {
      antigen_name: 'H5N1_TEXAS',
      gpu_device: '3',
      input_file_path: '/path/to/input',
    },
  },
  airr: {
    download_airr_sequences: {
      format: 'airr',
      max_sequences: 10000,
      output_dir: '/path/to/output',
      repertoire_id: '5ed6859e99011334ac05e847',
    },
    filter_by_vdj_genes: {
      combination_logic: 'AND',
      d_gene: 'example_d_gene',
      j_gene: 'example_j_gene',
      repertoire_id: '5ed6859e99011334ac05e847',
      v_gene: 'example_v_gene',
    },
    get_airr_statistics: {
      metrics: [],
      repertoire_id: '5ed6859e99011334ac05e847',
    },
    get_airr_study_metadata: {
      repository: 'auto',
      study_id: 'PRJNA300878',
    },
    search_airr_repertoires: {
      cell_subset: 'memory',
      disease: 'COVID-19',
      max_results: 100,
      repository: 'all',
      species: 'human',
      tissue: 'peripheral blood',
    },
  },
  anarci: {
    number_antibody_batch: {
      assign_germline: true,
      scheme: 'chothia',
      sequences: [
      ],
    },
    number_single_sequence: {
      scheme: 'imgt',
      sequence: 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWINWVRQAPGQGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
    },
  },
  annotation: {
    annotate_by_markers: {
      cluster_column: 'seurat_clusters',
      input_rds: 'example_input_rds',
      new_column: 'manual_celltype',
    },
    detect_cluster_markers: {
      input_rds: 'example_input_rds',
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
      input_rds: 'example_input_rds',
    },
    run_singler_annotation: {
      cluster_column: 'seurat_clusters',
      input_rds: 'example_input_rds',
      label_type: 'label.main',
      reference_dataset: 'HumanPrimaryCellAtlasData',
    },
    score_annotation_confidence: {
      annotation_column: 'example_annotation_column',
      input_rds: 'example_input_rds',
    },
    validate_annotation: {
      annotation_column1: 'example_annotation_column1',
      annotation_column2: 'example_annotation_column2',
      input_rds: 'example_input_rds',
      reference_dataset: 'MonacoImmuneData',
    },
  },
  bcell: {
    analyze_affinity_maturation: {
      airr_file: '/path/to/input',
      input_rds: 'example_input_rds',
      pseudotime_column: 'pseudotime',
    },
    analyze_bcell_trajectory: {
      input_rds: 'example_input_rds',
      method: 'monocle3',
      root_state: 'Naive',
      use_partition: true,
    },
    analyze_clonal_expansion: {
      airr_file: '/path/to/input',
      subset_metadata: 'example_subset_metadata',
      top_n_clones: 150,
    },
    analyze_gc_dynamics: {
      gc_subsets: ['DZ_GC', 'LZ_GC'],
      input_rds: 'example_input_rds',
      plot_markers: true,
    },
    analyze_isotype_distribution: {
      bcr_data: 'example_bcr_data',
      input_rds: 'example_input_rds',
      plot_type: 'bar',
    },
    analyze_vdj_usage: {
      airr_file: '/path/to/input',
      plot_heatmap: true,
      subset_metadata: 'example_subset_metadata',
    },
    calculate_repertoire_diversity: {
      airr_file: '/path/to/input',
      grouping_column: 'subset',
      metrics: ['shannon', 'simpson', 'hill'],
    },
    calculate_shm_rates: {
      airr_file: '/path/to/input',
      input_rds: 'example_input_rds',
      min_sequences: 10,
      region: 'IGHV',
    },
    compare_rsv_a_vs_b_bcells: {
      input_rds: 'example_input_rds',
      logfc_threshold: 0.25,
      min_pct: 0.1,
      rsv_metadata: 'example_rsv_metadata',
    },
    generate_bcell_report: {
      airr_file: '/path/to/input',
      cytotrace_rds: 'example_cytotrace_rds',
      input_rds: 'example_input_rds',
      report_format: 'html',
      trajectory_rds: 'example_trajectory_rds',
    },
    identify_antigen_specific_bcells: {
      binding_predictions: 'example_binding_predictions',
      input_rds: 'example_input_rds',
      min_binding_score: 0.5,
    },
    identify_bcell_subsets: {
      input_rds: 'example_input_rds',
      resolution: 0.8,
      species: 'human',
    },
    identify_convergent_sequences: {
      airr_file: '/path/to/input',
      cdr3_similarity_threshold: 0.85,
      min_group_size: 2,
    },
    predict_plasma_potential: {
      gene_signatures: 'example_gene_signatures',
      input_rds: 'example_input_rds',
    },
    run_cytotrace_analysis: {
      enable_fast: true,
      input_rds: 'example_input_rds',
      ncores: 4,
      subsample_size: 1000,
    },
  },
  bioinformatics: {
    antigen_binding_neutralization_density_visualization: {
      base_dir: '/path/to/input',
      feature_priority: 'neutralization_first',
      input_file: '/path/to/input',
      na_strategy: 'exclude_cells',
      prediction_keywords: 'neut,bind,average,predict,output',
    },
    antigen_binding_prediction_visualization: {
      base_dir: '/path/to/input',
      binding_threshold: 0.5,
      input_file: '/path/to/input',
    },
    bcell_celltype_distribution_analysis: {
      base_dir: '/path/to/input',
      input_file: '/path/to/input',
    },
    bcell_celltype_umap_visualization: {
      base_dir: '/path/to/input',
      celltype_column: 'CellType',
      input_file: '/path/to/input',
    },
    bcell_marker_gene_dotplot_analysis: {
      base_dir: '/path/to/input',
      input_file: '/path/to/input',
      min_expression: 0.25,
      min_pct: 0.1,
    },
    bcell_marker_gene_expression_dotplot: {
      base_dir: '/path/to/input',
      celltype_column: 'CellType',
      input_file: '/path/to/input',
    },
    bcr_isotype_distribution_shm_analysis: {
      base_dir: '/path/to/input',
      binding_threshold: 0.5,
      input_file: '/path/to/input',
    },
    binding_prediction_interval_distribution_analysis: {
      base_dir: '/path/to/input',
      data_max: 1.0,
      data_min: 0.0,
      input_file: '/path/to/input',
      interval_step: 0.1,
    },
    differential_gene_correlation_analysis: {
      base_dir: '/path/to/input',
      dataset1_name: 'example_dataset1_name',
      dataset2_name: 'example_dataset2_name',
      deg_file1: '/path/to/input',
      deg_file2: '/path/to/input',
      highlight_genes: 'ITGAX,FGR,FCRL4,FCRL5,CD68,TNFRSF1B,JCHAIN,MZB1,XBP1,MARCKSL1',
      min_common_genes: 10,
      p_value_threshold: 0.05,
    },
    differential_gene_expression_volcano_analysis: {
      analysis_strategy: 'both',
      base_dir: '/path/to/input',
      input_file: '/path/to/input',
      logfc_threshold: 0.0,
      min_pct: 0.2,
    },
    neutralizing_antibody_shm_comparison_analysis: {
      base_dir: '/path/to/input',
      binding_threshold: 0.5,
      input_file: '/path/to/input',
    },
    prediction_value_density_visualization: {
      base_dir: '/path/to/input',
      input_file: '/path/to/input',
      prediction_keywords: 'bind,predict,output,average,score',
      prediction_threshold: 0.5,
    },
    pseudotime_celltype_boxplot_analysis: {
      base_dir: '/path/to/input',
      celltype_column: '',
      input_file: '/path/to/input',
    },
    pseudotime_trajectory_analysis: {
      base_dir: '/path/to/input',
      cluster_resolution: 0.001,
      input_file: '/path/to/input',
      min_gene_cells: 3,
      num_dim: 50,
      root_celltype: 'Naive',
    },
    trajectory_polynomial_regression_analysis: {
      base_dir: '/path/to/input',
      input_file: '/path/to/input',
    },
    trajectory_supplementary_analysis: {
      base_dir: '/path/to/input',
      input_file: '/path/to/input',
    },
    umap_dimensionality_reduction_visualization: {
      base_dir: '/path/to/input',
      input_file: '/path/to/input',
    },
  },
  communication: {
    analyze_signaling_pathways: {
      cellchat_rds: 'example_cellchat_rds',
      output_prefix: 'pathways',
      pathways: 'all',
    },
    calculate_network_centrality: {
      cellchat_rds: 'example_cellchat_rds',
      output_prefix: 'centrality',
    },
    compare_communication_networks: {
      cellchat_list: 'example_cellchat_list',
      group_names: 'example_group_names',
      output_prefix: 'comparison',
    },
    identify_ligand_receptor_pairs: {
      group_by: 'celltype',
      input_rds: 'example_input_rds',
      min_pct: 0.1,
      output_prefix: 'lr_pairs',
      species: 'mouse',
    },
    plot_communication_network: {
      cellchat_rds: 'example_cellchat_rds',
      layout: 'circle',
      output_prefix: 'network',
      signaling: 'example_signaling',
    },
    run_cellchat_analysis: {
      db_type: 'Secreted Signaling',
      group_by: 'celltype',
      input_rds: 'example_input_rds',
      output_prefix: 'cellchat',
      species: 'mouse',
    },
    run_nichenet_analysis: {
      condition_colname: 'orig.ident',
      condition_oi: 'example_condition_oi',
      condition_reference: 'example_condition_reference',
      input_rds: 'example_input_rds',
      output_prefix: 'nichenet',
      receiver_cells: 'example_receiver_cells',
      sender_cells: 'example_sender_cells',
    },
    run_spatial_communication: {
      contact_range: 100.0,
      group_by: 'celltype',
      input_rds: 'example_input_rds',
      output_prefix: 'spatial_comm',
      spatial_coordinates: 'example_spatial_coordinates',
      species: 'mouse',
    },
  },
  fdg: {
    fdg: {
      input_file_path: '/path/to/input',
    },
  },
  geo: {
    download_geo_sequences: {
      data_format: 'processed',
      geo_id: 'example_id_123',
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
      sequences: [
      ],
    },
    get_all_geo_bcr_studies: {
      include_metadata: true,
      max_results: 1000,
      organism: 'Homo sapiens',
    },
    get_geo_metadata: {
      geo_id: 'example_id_123',
      include_samples: true,
    },
    get_iedb_antibody_data: {
      antibody_id: 'example_id_123',
      antibody_name: 'example_antibody_name',
      include_structures: true,
    },
    list_geo_bcr_datasets: {
      curated_only: true,
    },
    map_cdr3_to_epitopes: {
      cdr3_list: ['item1', 'item2', 'item3'],
      confidence_threshold: 0.7,
      disease_context: 'COVID-19',
    },
    search_geo_studies: {
      keywords: 'example_keywords',
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
      cdr3_sequences: [
        {
          id: 'seq1',
          sequence: 'ATGGAGTTTGGGCTGAGCTGGGTTTTCCTTGTTGCTATTTTAAAAGGTGTCCAGTGTGAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTCAGTAGCTATGCTATGCACTGGGTCCGCCAGGCTCCAGGCAAGGGGCTGGAGTGGGTGGCAGTTATATCATATGATGGAAGTAATAAATACTATGCAGACTCCGTGAAGGGCCGATTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCTGTGTATTACTGTGCGAGAGA',
        },
        {
          id: 'seq2',
          sequence: 'ATGGAGTTTGGGCTGAGCTGGGTTTTCCTTGTTGCTATTTTAAAAGGTGTCCAGTGTGAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTCAGTAGCTATGCTATGCACTGGGTCCGCCAGGCTCCAGGCAAGGGGCTGGAGTGGGTGGCAGTTATATCATATGATGGAAGTAATAAATACTATGCAGACTCCGTGAAGGGCCGATTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCTGTGTATTACTGTGCGAGAGA',
        },
        {
          id: 'seq3',
          sequence: 'ATGGAGTTTGGGCTGAGCTGGGTTTTCCTTGTTGCTATTTTAAAAGGTGTCCAGTGTGAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTCAGTAGCTATGCTATGCACTGGGTCCGCCAGGCTCCAGGCAAGGGGCTGGAGTGGGTGGCAGTTATATCATATGATGGAAGTAATAAATACTATGCAGACTCCGTGAAGGGCCGATTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCTGTGTATTACTGTGCGAGAGA',
        },
      ],
      epitope_sequence: 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWINWVRQAPGQGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
    },
  },
  lgblast: {
    analyze_vdj_batch: {
      locus: 'IGH',
      organism: 'human',
      receptor_type: 'Ig',
      sequences: [],
    },
    extract_cdr3_from_airr: {
      airr_results: [
      ],
    },
  },
  metabcr: {
    metabcr: {
      input_file_path: '/path/to/input',
      output_file_path: '/path/to/output',
    },
  },
  multimodal: {
    identify_multimodal_clusters: {
      algorithm: 3,
      resolution: 0.8,
      wnn_rds: 'example_wnn_rds',
    },
    integrate_multimodal_wnn: {
      adt_dims: 18,
      adt_rds: 'example_adt_rds',
      atac_dims: 30,
      atac_rds: 'example_atac_rds',
      rna_dims: 30,
      rna_rds: 'example_rna_rds',
    },
    link_peaks_to_genes: {
      atac_rds: 'example_atac_rds',
      distance: 500000,
      min_correlation: 0.05,
      rna_rds: 'example_rna_rds',
    },
    plot_multimodal_features: {
      features: ['item1', 'item2', 'item3'],
      modalities: [],
      plot_type: 'umap',
      wnn_rds: 'example_wnn_rds',
    },
    process_adt_data: {
      adt_counts_file: '/path/to/input',
      min_cells: 3,
      min_features: 3,
      normalization_method: 'CLR',
    },
    process_atac_data: {
      fragments_file: '/path/to/input',
      genome: 'hg38',
      max_fragments: 100000,
      min_cells: 10,
      min_fragments: 1000,
      peaks_file: '/path/to/input',
      tss_enrichment_threshold: 2.0,
    },
  },
  oas: {
    batch_process_oas: {
      batch_size: 100000,
      operation: 'example_operation',
      parallel: false,
      study_ids: ['item1', 'item2', 'item3'],
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
      cdr3_pattern: 'example_cdr3_pattern',
      j_gene: 'example_j_gene',
      min_identity: 0.0,
      study_id: 'PRJNA300878',
      v_gene: 'example_v_gene',
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
  scrna: {
    run_clustering_analysis: {
      algorithm: 'leiden',
      dims: 30,
      input_rds: 'example_input_rds',
      resolution: 0.5,
    },
    run_deg_analysis: {
      group_by: 'seurat_clusters',
      ident_1: 'example_id_123',
      ident_2: 'example_id_123',
      input_rds: 'example_input_rds',
      logfc_threshold: 0.25,
      min_pct: 0.1,
      test_use: 'wilcox',
    },
    run_dim_reduction: {
      dims: 30,
      input_rds: 'example_input_rds',
      methods: [],
      min_dist: 0.3,
      n_neighbors: 30,
    },
    run_doublet_detection: {
      dims: 20,
      expected_doublet_rate: 0.08,
      input_rds: 'example_input_rds',
      pK: 0.09,
      pN: 0.25,
    },
    run_full_preprocessing_pipeline: {
      dims: 30,
      input_rds: 'example_input_rds',
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
      input_rds: 'example_input_rds',
      theta: [],
    },
    run_marker_detection: {
      group_by: 'seurat_clusters',
      input_rds: 'example_input_rds',
      logfc_threshold: 0.5,
      min_pct: 0.25,
      only_pos: true,
      top_n: 10,
    },
    run_normalization_sct: {
      input_rds: 'example_input_rds',
      n_variable_features: 3000,
      vars_to_regress: [],
    },
    run_pathway_enrichment: {
      deg_csv: 'example_deg_csv',
      input_rds: 'example_input_rds',
      ontology: 'BP',
      organism: 'human',
      pvalue_cutoff: 0.05,
      qvalue_cutoff: 0.2,
    },
    run_qc_filtering: {
      input_rds: 'example_input_rds',
      max_genes: 6000,
      min_counts: 1000,
      min_genes: 200,
      mt_percent: 20.0,
    },
    run_subset_cells: {
      input_rds: 'example_input_rds',
      invert: false,
      subset_column: 'example_subset_column',
      subset_values: ['item1', 'item2', 'item3'],
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
