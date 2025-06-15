import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ----------------------------------------加载微调模型-------------------------------------------#
# 基础模型路径
base_model_path = "./dataroot/models/Qwen/Qwen3-0.6B"
# LoRA适配器路径
lora_adapter_path = "Qwen3-0.6B-LoRA-Racism"

# 检查模型路径是否存在
if not os.path.exists(base_model_path):
    print(f"错误: 基础模型路径不存在 {base_model_path}")
    exit(1)
if not os.path.exists(lora_adapter_path):
    print(f"错误: LoRA适配器路径不存在 {lora_adapter_path}")
    exit(1)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True
)

# 加载基础模型
try:
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype="auto",
        local_files_only=True
    ).to(device)
    print("基础模型加载完成")
except Exception as e:
    print(f"加载基础模型失败: {str(e)}")
    exit(1)

# 加载LoRA适配器
try:
    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path).to(device)

    # 合并LoRA权重到基础模型
    print("合并LoRA权重...")
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    else:  # 兼容旧版本
        model = model.merge_adapter()

    model.eval()  # 设置为评估模式
    print("微调模型加载完成，LoRA权重已合并")
except Exception as e:
    print(f"加载LoRA适配器失败: {str(e)}")
    exit(1)


# ------------------------------加载微调模型-------------------------------------------#

# -------------------------------定义生成响应的函数-------------------------------------#
def generate_response(prompt, model, max_new_tokens=256):
    """
    生成仇恨检测响应
    :param prompt: 输入提示
    :param model: 模型实例
    :param max_new_tokens: 最大生成token数
    :return: 生成的文本
    """
    try:
        # 对输入进行分词
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(device)

        # 生成响应
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 计算生成时间
        gen_time = time.time() - start_time

        # 解码生成的文本（跳过输入部分）
        response = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        # 清理输出：去除多余的空格和换行
        return response.split("\n")[0].strip(), gen_time
    except Exception as e:
        print(f"生成响应时出错: {str(e)}")
        return f"[生成失败: {str(e)}]", 0


# -------------------------------定义生成响应的函数-------------------------------------#

# -------------------------------处理测试数据集函数-------------------------------------#
def process_test_dataset(test_file):
    """
    处理测试数据集并生成仇恨检测结果
    :param test_file: 测试数据集文件路径
    :return: 包含结果的数据列表
    """
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"错误: 测试文件不存在 {test_file}")
        return []

    # 加载测试数据集
    try:
        print(f"加载测试数据集: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"数据集加载完成，共 {len(test_data)} 条数据")
    except Exception as e:
        print(f"加载测试数据失败: {str(e)}")
        return []

    results = []
    total_gen_time = 0

    # 逐条处理数据
    print("开始生成仇恨检测结果...")
    for item in tqdm(test_data, desc="处理进度"):
        try:
            # 检查必需字段是否存在
            if "instruction" not in item:
                raise KeyError("缺少instruction字段")
            if "content" not in item:
                raise KeyError("缺少content字段")

            # 获取ID
            item_id = item.get("id", "N/A")

            # 构建输入提示
            prompt = (
                    item["instruction"] +
                    "\n\n我需要你进行类似上述处理的句子是:" +
                    item["content"] +
                    "\n\n"
            )

            # 生成模型响应
            generated, gen_time = generate_response(prompt, model)
            total_gen_time += gen_time

            # 保存结果 - 严格遵循输出格式
            result_item = {
                "id": item_id,
                "content": item["content"],
                "output": generated
            }
            results.append(result_item)

        except Exception as e:
            error_msg = str(e)
            print(f"处理ID {item.get('id', 'unknown')} 时出错: {error_msg}")
            results.append({
                "id": item.get("id", "error"),
                "content": item.get("content", "[无内容]"),
                "output": f"[处理失败: {error_msg}]"
            })

    # 打印性能统计
    if results:
        avg_time = total_gen_time / len(results)
        print(f"处理完成: 共 {len(results)} 条数据")
        print(f"平均生成时间: {avg_time:.2f}秒/条")

    return results


# -------------------------------处理测试数据集函数-------------------------------------#

# -------------------------------主执行函数-------------------------------------#
def main():
    # 定义测试数据集
    test_files = [
        "test1_with_instruction.json",
        # "test2_with_instruction.json"
    ]

    # 处理所有测试数据集
    all_results = {}
    for test_file in test_files:
        print(f"\n{'=' * 50}")
        print(f"开始处理: {test_file}")
        results = process_test_dataset(test_file)
        all_results[test_file] = results
        print(f"完成处理: {test_file}")
        print(f"{'=' * 50}\n")

    # 将结果保存为严格的JSON格式文件
    try:
        # 创建最终结果字典
        final_output = {}

        # 为每个数据集创建结果文件
        for file_name, results in all_results.items():
            # 创建输出文件名（保持原始文件名前缀）
            output_file = file_name.replace("_with_instruction.json", "_results.json")

            # 保存为JSON文件
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"已保存结果到: {output_file}")

            # 添加到最终输出字典
            final_output[file_name] = {
                "output_file": output_file,
                "count": len(results)
            }

        # 保存汇总信息
        with open("results_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print("\n所有测试已完成！")
        print(f"总条目数: {sum(len(res) for res in all_results.values())}")
        print(f"结果摘要已保存到: results_summary.json")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")


# -------------------------------主执行函数-------------------------------------#

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"总执行时间: {total_time:.2f}秒")