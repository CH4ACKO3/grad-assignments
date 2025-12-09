import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_boston_housing():
    """Load Boston Housing dataset from original source"""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    
    return data, target, feature_names

def xgboost_boston_housing():
    """Build XGBoost model on Boston Housing data with 3-fold CV"""
    
    # Load Boston Housing dataset
    X, y, feature_names = load_boston_housing()
    
    print("Boston Housing Dataset")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {list(feature_names)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Prepare data for XGBoost
    dtrain = xgb.DMatrix(X, label=y)
    
    # 3-fold cross-validation using xgb.cv()
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        nfold=3,
        metrics=['rmse', 'mae'],
        seed=42,
        shuffle=True,
        as_pandas=True,
        verbose_eval=False
    )
    
    print("3-Fold Cross-Validation Results (using xgb.cv):")
    print(cv_results)
    print()
    print(f"Final CV RMSE: {cv_results['test-rmse-mean'].iloc[-1]:.4f} (+/- {cv_results['test-rmse-std'].iloc[-1]:.4f})")
    print(f"Final CV MAE: {cv_results['test-mae-mean'].iloc[-1]:.4f} (+/- {cv_results['test-mae-std'].iloc[-1]:.4f})")
    print()
    
    # Save CV results to CSV for LaTeX
    cv_results.to_csv('result/xgboost_cv_results.csv', index=False)
    print("Saved CV results to result/xgboost_cv_results.csv")
    
    # Fit on full data for feature importance and tree visualization
    xgb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100
    )
    
    # Feature importance
    importance_dict = xgb_model.get_score(importance_type='weight')
    # Convert to list format matching feature names
    feature_importance_list = []
    for i, name in enumerate(feature_names):
        # XGBoost uses f0, f1, f2, ... as feature names
        feature_key = f'f{i}'
        importance = importance_dict.get(feature_key, 0)
        feature_importance_list.append(importance)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_list
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance.to_string(index=False))
    print()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('XGBoost Feature Importance (Boston Housing)', fontsize=14)
    plt.tight_layout()
    plt.savefig('result/xgboost_feature_importance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved feature importance plot")
    
    # Plot tree (first tree)
    try:
        # Try using graphviz for better visualization
        import graphviz
        import re
        # Create graphviz source with custom parameters for better layout
        graph_source = xgb.to_graphviz(xgb_model, num_trees=0)
        
        # Get the source string and modify it to add custom attributes
        source_str = graph_source.source
        
        # Find the graph declaration line and add custom attributes
        # Replace the default graph declaration with one that has better layout
        if source_str.startswith('digraph'):
            # Add custom attributes after the graph declaration
            # Insert attributes before the first node/edge definition
            lines = source_str.split('\n')
            new_lines = []
            graph_declared = False
            root_node = None
            
            # Find the root node (node with no incoming edges)
            # In XGBoost trees, root is typically node 0, but we'll detect it properly
            source_nodes = set()
            target_nodes = set()
            
            for line in lines:
                line_stripped = line.strip()
                if '->' in line_stripped and not line_stripped.startswith('//'):
                    # Parse edge: "source -> target" or "source -> target [label=...]"
                    edge_part = line_stripped.split('[')[0] if '[' in line_stripped else line_stripped
                    if '->' in edge_part:
                        parts = edge_part.split('->')
                        if len(parts) == 2:
                            source = parts[0].strip().strip('"').strip("'")
                            target = parts[1].strip().strip('"').strip("'")
                            source_nodes.add(source)
                            target_nodes.add(target)
            
            # Root node is one that is a source but never as target
            potential_roots = source_nodes - target_nodes
            if potential_roots:
                # Prefer node "0" if it exists, otherwise take any root
                if '0' in potential_roots:
                    root_node = '0'
                else:
                    root_node = list(potential_roots)[0]
            else:
                # Fallback: assume node 0 is root (common in XGBoost)
                root_node = '0'
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # 1. 处理 digraph 声明行
                if line_stripped.startswith('digraph') and not graph_declared:
                    new_lines.append(line) # Append the original digraph line
                    
                    # Add custom graph attributes immediately after the digraph declaration
                    # Ensure rankdir=LR is added here and is not overridden
                    new_lines.append('  rankdir=LR;')  # Left to right (landscape)
                    new_lines.append('  size="8.27,11.69";')  # A4 landscape: width,height in inches (宽,高)
                    new_lines.append('  dpi=300;')  # High resolution
                    new_lines.append('  nodesep=0.1;')  # Space between nodes (horizontal)
                    new_lines.append('  ranksep=0.3;')  # Space between levels (vertical)
                    new_lines.append('  splines=true;') # Orthogonal edges
                    new_lines.append('  overlap=scale;') # Prevent nodes from overlapping
                    new_lines.append('  ratio=fill;')  # Fill the page
                    new_lines.append('  ordering=out;')  # Ensure proper ordering from root
                    new_lines.append('  node [fontsize=20, shape=box, style="filled,rounded", fillcolor="white", margin="0.2,0.1"];')
                    graph_declared = True
                    continue # Skip to next line, as we've handled this one
                
                # 2. 检查并跳过任何可能存在的 rankdir 属性，避免冲突
                if graph_declared:
                    if line_stripped.startswith('rankdir='):
                        print(f"DEBUG: Skipping existing rankdir declaration: {line_stripped}")
                        continue
                    match_graph_attr = re.match(r'^\s*graph\s*\[(.*)\]\s*$', line_stripped)
                    if match_graph_attr and 'rankdir=' in match_graph_attr.group(1):
                        print(f"DEBUG: Skipping graph attribute block containing rankdir: {line_stripped}")
                        continue
                    if re.match(r'^\s*node\s*\[(.*)\]\s*$', line_stripped):
                        print(f"DEBUG: Skipping existing global node attribute block: {line_stripped}")
                        continue
                # --- 3. 处理节点定义行，并格式化小数位数 ---
                # 匹配节点定义行，例如 "0 [label="...", ...]"
                # 注意：这里需要更通用的匹配，因为节点号可以是数字或带引号的字符串
                node_def_match = re.match(r'^\s*(\w+|\"\w+\"|\'\w+\')\s*\[(.*)\]\s*$', line_stripped)
                
                if node_def_match:
                    node_id = node_def_match.group(1)
                    attributes_str = node_def_match.group(2)
                    
                    # 提取 label 属性
                    label_match = re.search(r'label="([^"]*)"', attributes_str)
                    if label_match:
                        original_label_text = label_match.group(1)
                        
                        # 定义一个函数来替换浮点数
                        def format_float_in_label(match):
                            try:
                                formatted_num = f"{float(match.group(0)):.2f}" 
                                return formatted_num
                            except ValueError:
                                return match.group(0)
                        
                        # 使用正则表达式查找并替换标签中的浮点数
                        float_pattern = r'(-?\d+\.\d+)' 
                        new_label_text = re.sub(float_pattern, format_float_in_label, original_label_text)
                        
                        # 重新构建 label 属性
                        new_attributes_str = re.sub(r'label="[^"]*"', f'label="{new_label_text}"', attributes_str)
                        
                        # 如果是根节点，还需要处理 rank=source 的插入
                        if root_node and (node_id == root_node or node_id.strip('"\'') == root_node):
                            if 'rank=source' not in new_attributes_str and 'rank=' not in new_attributes_str:
                                if new_attributes_str.endswith(';'):
                                    new_attributes_str = new_attributes_str.rstrip(';') + ' rank=source;'
                                else:
                                    new_attributes_str = new_attributes_str.strip() + ' rank=source'
                        
                        # 重新构建整行
                        indent = len(line) - len(line.lstrip())
                        new_lines.append(' ' * indent + f'{node_id} [{new_attributes_str.strip()}];')
                    else:
                        new_lines.append(line)
                elif root_node and (line_stripped.startswith(f'{root_node} [') or line_stripped.startswith(f'"{root_node}" [') or line_stripped.startswith(f"'{root_node}' [")):
                    if 'rank=source' not in line_stripped and 'rank=' not in line_stripped:
                        if ']' in line_stripped:
                            line_parts = line_stripped.rsplit(']', 1)
                            attr_part = line_parts[0]
                            if attr_part.endswith(';'):
                                new_attr = attr_part.rstrip(';') + ' rank=source;'
                            else:
                                new_attr = attr_part + ' rank=source'
                            new_line = new_attr + ']' + line_parts[1]
                            indent = len(line) - len(line.lstrip())
                            new_lines.append(' ' * indent + new_line)
                        else: 
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            modified_source = '\n'.join(new_lines)
        else:
            modified_source = source_str
        
        # Create graph with modified source
        graph = graphviz.Source(modified_source, format='pdf', engine='dot')
        
        # Render to PDF with high quality
        graph.render('result/xgboost_tree', cleanup=True)
        print("Saved tree visualization as PDF using graphviz (landscape A4, left-to-right, high resolution)")
    except ImportError:
        print("graphviz not available, trying matplotlib...")
        try:
            # Use landscape A4 orientation with higher resolution
            # A4 landscape: 11.69 x 8.27 inches
            fig, ax = plt.subplots(figsize=(11.69, 8.27))  # Landscape A4: width=11.69, height=8.27
            xgb.plot_tree(xgb_model, num_trees=0, ax=ax)
            plt.title('XGBoost Tree Visualization (First Tree)', fontsize=14)
            plt.tight_layout()
            plt.savefig('result/xgboost_tree.pdf', dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            print("Saved tree visualization using matplotlib (landscape A4, high resolution)")
        except Exception as e:
            print(f"Could not plot tree with matplotlib: {e}")
            # Alternative: save as text
            tree_str = xgb_model.get_booster().get_dump()[0]
            with open('result/xgboost_tree.txt', 'w') as f:
                f.write(tree_str)
            print("Saved tree as text file")
    except Exception as e:
        print(f"Could not plot tree: {e}")
        # Alternative: save as text
        tree_str = xgb_model.get_booster().get_dump()[0]
        with open('result/xgboost_tree.txt', 'w') as f:
            f.write(tree_str)
        print("Saved tree as text file")
    
    return xgb_model, cv_results

if __name__ == "__main__":
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    
    # Change to script directory to ensure relative paths work correctly
    original_cwd = os.getcwd()
    os.chdir(script_dir)
    
    try:
        model, cv_results = xgboost_boston_housing()
        print("\nXGBoost model training completed!")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
