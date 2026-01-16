from topicgpt_python import *
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

def run_inference():
    generate_topic_lvl1(
        "ollama",
        "gpt-oss",
        config["data_sample"],
        config["generation"]["prompt"],
        config["generation"]["seed"],
        config["generation"]["output"],
        config["generation"]["topic_output"],
        verbose=config["verbose"],
    )

    # Optional: Refine topics if needed
# if config["refining_topics"]:
#     refine_topics(
#         "ollama",
#         "gpt-oss",
#         config["refinement"]["prompt"],
#         config["generation"]["output"],
#         config["generation"]["topic_output"],
#         config["refinement"]["topic_output"],
#         config["refinement"]["output"],
#         verbose=config["verbose"],
#         remove=config["refinement"]["remove"],
#         mapping_file=config["refinement"]["mapping_file"]
#     )

    # Assignment
    assign_topics(
        "ollama",
        "gpt-oss",
        config["data_sample"],
        config["assignment"]["prompt"],
        config["assignment"]["output"],
        config["generation"][
            "topic_output"
        ],  # TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
        verbose=config["verbose"],
    )

    # Correction
    correct_topics(
        "ollama",
        "gpt-oss",
        config["assignment"]["output"],
        config["correction"]["prompt"],
        config["generation"][
            "topic_output"
        ],  # TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
        config["correction"]["output"],
        verbose=config["verbose"],
    )

if __name__ == "__main__":
    run_inference()