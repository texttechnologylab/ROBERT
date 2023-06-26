db.getCollection("chatgpt_scores").aggregate([
    {
        // Use "$eq" for instruction following, use "$ne" for dialog checking
        "$match": {"input": {"$eq": ""}}
    },
     {
       "$group":
         {
           _id: "$model",
           chatgptscore: {"$avg": "$score"},
           bleu: { "$avg": "$rogue_score.bleu" },
           rouge1_precision: { "$avg": "$rogue_score.rouge1_precision" },
           rouge1_fmeasure: { "$avg": "$rogue_score.rouge1_fmeasure" },
           rouge2_precision: { "$avg": "$rogue_score.rouge2_precision" },
           rouge2_fmeasure: { "$avg": "$rogue_score.rouge2_fmeasure" },
           rougeL_precision: { "$avg": "$rogue_score.rougeL_precision" },
           rougeL_fmeasure: { "$avg": "$rogue_score.rougeL_fmeasure" },
           rougeLsum_precision: { "$avg": "$rogue_score.rougeLsum_precision" },
           rougeLsum_fmeasure: { "$avg": "$rogue_score.rougeLsum_fmeasure" }
         }
     }
])
