db.getCollection("rouge_scores").aggregate([
    {
        // Use "$eq" for instruction following, use "$ne" for dialog checking
        "$match": {"inp": {"$eq": ""}}
    },
     {
       "$group":
         {
           _id: "$model",
           bleu: { "$avg": "$bleu" },
           rouge1_precision: { "$avg": "$rouge1_precision" },
           rouge1_fmeasure: { "$avg": "$rouge1_fmeasure" },
           rouge2_precision: { "$avg": "$rouge2_precision" },
           rouge2_fmeasure: { "$avg": "$rouge2_fmeasure" },
           rougeL_precision: { "$avg": "$rougeL_precision" },
           rougeL_fmeasure: { "$avg": "$rougeL_fmeasure" },
           rougeLsum_precision: { "$avg": "$rougeLsum_precision" },
           rougeLsum_fmeasure: { "$avg": "$rougeLsum_fmeasure" },
           bleu: { "$avg": "$bleu" }           
         }
     }
])
