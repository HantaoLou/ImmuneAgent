import React, { useState } from 'react';
import { ChevronDown, ChevronRight, FileText } from 'lucide-react';
import { Template } from '@/lib/types';

interface TemplatePanelProps {
  templates: Template[];
  onTemplateSelect: (template: Template) => void;
}

export function TemplatePanel({ templates, onTemplateSelect }: TemplatePanelProps) {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['General'])
  );

  const categories = [...new Set(templates.map((t) => t.category))];

  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  return (
    <div className="border-t border-gray-200">
      <div className="p-3 bg-gray-50">
        <h3 className="text-sm font-semibold text-gray-700">Templates</h3>
      </div>
      <div className="overflow-y-auto max-h-64">
        {categories.map((category) => (
          <div key={category}>
            <button
              onClick={() => toggleCategory(category)}
              className="w-full flex items-center justify-between p-2 hover:bg-gray-50 text-sm font-medium text-gray-700"
            >
              <span>{category}</span>
              {expandedCategories.has(category) ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </button>
            {expandedCategories.has(category) && (
              <div className="pl-4">
                {templates
                  .filter((t) => t.category === category)
                  .map((template) => (
                    <button
                      key={template.id}
                      onClick={() => onTemplateSelect(template)}
                      className="w-full flex items-center space-x-2 p-2 hover:bg-blue-50 text-left text-sm text-gray-600"
                    >
                      <FileText className="h-3 w-3" />
                      <span className="truncate">{template.name}</span>
                    </button>
                  ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
